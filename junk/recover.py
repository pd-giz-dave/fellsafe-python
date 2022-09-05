class Scan:

    def recover(self, target):

        def make_edge(transitions, direction):
            """ make the best continuous edge from the given transitions """

            # for an X
            #  for an edge
            #    connect closest for full circle - save with its metrics
            #  repeat for next edge
            # repeat for next X
            # pick best

            def make_candidate(start_x, start_y):
                """ make a candidate edge from transitions from start_x/y,
                    edge pairs at x and x+1 and x and x+2 are considered,
                    the next y is the closest of those 2 pairs
                    """

                def get_nearest_y(x, y):
                    """ find the y with the minimum gap and strongest edge to the given y at slice x,
                        for any x,y there are 3 immediate neighbours: (x+1,y-1),(x+1,y),(x+1,y+1),
                        if more than one of these is the nearest neighbour we choose the one with the
                        'strongest' edge, 'strongest' is that with the pixel value closest to black
                        """
                    neighbours = []
                    min_gap = max_y * max_y
                    min_y = None
                    for t, transition in enumerate(transitions[x]):
                        gap = y - transition[0]
                        gap *= gap
                        if gap < min_gap:
                            min_gap = gap
                            min_y = transition[0]
                        if gap > Scan.MAX_NEIGHBOUR_GAP:
                            # not a neighbour
                            continue
                        neighbours.append((transition[0], transition[1], gap))
                    if len(neighbours) > 1:
                        # got a choice, pick the best
                        best = neighbours[0]
                        for n, neighbour in enumerate(neighbours):
                            if neighbour[1] < best[1]:
                                best = neighbour
                        return best[0], best[2]
                    else:
                        return min_y, min_gap

                candidate = [None for _ in range(max_x)]
                x = start_x - 1
                y = start_y
                for _ in range(max_x):
                    x = (x + 1) % max_x
                    this_y, this_gap = get_nearest_y(x, y)
                    if this_gap > Scan.MAX_NEIGHBOUR_GAP:
                        # direct neighbour not connected, see if indirect one is nearer
                        next_y, next_gap = get_nearest_y((x + 1) % max_x, y)
                        if this_gap <= next_gap:
                            # nope
                            y = this_y
                        else:
                            # yep
                            y = next_y
                    else:
                        # direct neighbour connected, use that
                        y = this_y
                    candidate[x] = y

                return candidate

            # This makes lots of duplicates, but it is easier to deal with them than suppress them!
            edges = []
            for x in range(max_x):
                for transition in transitions[x]:
                    edges.append(make_candidate(x, transition[0]))

            # pick best
            # ToDo: improve this to allow for big jumps out then in - interpolate across the gap
            best_edge = None
            best_jumps = None
            for edge in edges:
                # count y jumps
                jumps = 0
                for x in range(max_x):
                    last_y = edge[(x - 1) % max_x]
                    this_y = edge[x]
                    gap = last_y - this_y
                    gap *= gap
                    if gap > Scan.MAX_NEIGHBOUR_GAP:
                        # not connected
                        jumps += gap
                if best_edge is None or jumps < best_jumps:
                    best_edge = edge
                    best_jumps = jumps

            if self.logging:
                self._log('measure {}: found {} edges, best has {} jumps'.
                          format(direction, len(edges), best_jumps))

            return best_edge

        # make an image to detect our edges within
        max_x, max_y = target.size()
        edges: Frame = target.instance()
        edges.new(max_x, max_y, MIN_LUMINANCE)

        # build bucketised image (reduce to the 4 luminance levels described above)
        bucket_range = int(round((MAX_LUMINANCE - MIN_LUMINANCE) / len(Scan.EDGE_THRESHOLD_LEVELS)))
        thresholds: List[Scan.Threshold] = [None for _ in range(max_x)]
        for x in range(max_x):
            threshold = self._get_slice_threshold(target, x, max_x, 0, max_y, Scan.EDGE_THRESHOLD_LEVELS)
            for y in range(threshold.limit):
                pixel = self._get_threshold_pixel(target, x, y, threshold, bucket=bucket_range)
                edges.putpixel(x, y, pixel)
            thresholds[x] = threshold
        # remove islands - pixels completely surrounded by the same colour become that colour
        for x in range(max_x):
            for y in range(thresholds[x].limit):
                is_island = False
                colour = None
                for dx, dy in Scan.ISLAND_KERNEL:
                    pixel = edges.getpixel((x + dx) % max_x, y + dy)
                    if pixel is None:
                        # ignore pixels off the edge
                        continue
                    if colour is None:
                        colour = pixel
                        is_island = True
                    elif pixel != colour:
                        is_island = False
                        break
                if is_island:
                    edges.putpixel(x, y, colour)

        # build list of transitions to black (the first is probably the inner edge)
        to_black = [[] for _ in range(max_x)]  # falling edges
        for x in range(max_x):
            threshold = thresholds[x]
            last_pixel = edges.getpixel(x, 0)
            for y in range(1, threshold.limit):
                pixel = edges.getpixel(x, y)
                if pixel < last_pixel:
                    # falling edge, want y of highest luminance (i.e. previous y)
                    to_black[x].append((y - 1, pixel))
                last_pixel = pixel
            if len(to_black[x]) == 0:
                # this means the whole slice is the same intensity
                to_black[x] = [(0, 0)]
        inner = make_edge(to_black, Scan.TOP_DOWN)

        # adjust threshold limit to reflect the inner edge we discovered
        for x in range(max_x):
            # inner y is assumed to represent the two inner white rings,
            # so the limit for the outer is y/2 * num-rings * a fudge factor
            thresholds[x].limit = int(min((inner[x] / 2) * Scan.NUM_RINGS * Scan.RING_RADIUS_STRETCH, max_y))

        # build list of transitions to white (the last is probably the outer edge)
        to_white = [[] for _ in range(max_x)]  # rising edges
        for x in range(max_x):
            threshold = thresholds[x]
            last_pixel = edges.getpixel(x, 0)
            for y in range(inner[x] + 1, threshold.limit):
                pixel = edges.getpixel(x, y)
                if pixel > last_pixel:
                    # rising edge, want y of highest luminance (i.e. this y)
                    to_white[x].append((y, last_pixel))
                last_pixel = pixel
            if len(to_white[x]) == 0:
                # this means the whole slice is black
                to_white[x] = [(max(threshold.limit - 1, 0), 0)]
        outer = make_edge(to_white, Scan.BOTTOM_UP)

        # smooth the edges
        inner_edge = self._smooth_edge(inner)
        outer_edge = self._smooth_edge(outer)

        if self.save_images:
            y_limit = [max(thresholds[x].limit - 1, 0) for x in range(max_x)]
            plot = edges
            plot = self._draw_plots(plot, [[0, y_limit]], colour=Scan.RED)
            plot = self._draw_plots(plot, [[0, inner]], colour=Scan.GREEN)
            plot = self._draw_plots(plot, [[0, outer]], colour=Scan.BLUE)
            self._unload(plot, '03-edges')

            plot = target
            plot = self._draw_plots(plot, [[0, inner_edge]], None, Scan.GREEN)
            plot = self._draw_plots(plot, [[0, outer_edge]], None, Scan.GREEN)
            self._unload(plot, '04-wavy')

        return Scan.Extent(target=target, inner_edge=inner_edge, outer_edge=outer_edge, size=orig_radius)
