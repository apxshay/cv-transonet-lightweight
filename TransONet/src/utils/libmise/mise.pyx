# distutils: language = c++
cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libc.math cimport isnan, NAN
import numpy as np
import time


cdef struct Vector3D:
    int x, y, z


cdef struct Voxel:
    Vector3D loc
    unsigned int level
    bint is_leaf
    bint positive
    bint negative
    bint queued
    unsigned long children[2][2][2]


cdef struct GridPoint:
    Vector3D loc
    double value
    bint known


cdef inline unsigned long vec_to_idx(Vector3D coord, long resolution):
    cdef unsigned long idx
    idx = resolution * resolution * coord.x + resolution * coord.y + coord.z
    return idx


cdef class MISE:
    cdef vector[Voxel] voxels
    cdef vector[GridPoint] grid_points
    cdef vector[int] grid_point_hash
    cdef vector[long] active_voxels
    cdef readonly int resolution_0
    cdef readonly int depth
    cdef readonly double threshold
    cdef readonly int voxel_size_0
    cdef readonly int resolution
    cdef double time_set_values, time_subdivide
    cdef double time_dense_alloc, time_dense_write, time_dense_fill
    cdef long n_subdivided
    cdef vector[long] subdivisions_by_level

    def __cinit__(self, int resolution_0, int depth, double threshold):
        self.resolution_0 = resolution_0
        self.depth = depth
        self.threshold = threshold
        self.voxel_size_0 = (1 << depth)
        self.resolution = resolution_0 * self.voxel_size_0
        self.subdivisions_by_level.resize(depth + 1, 0)
        self.grid_point_hash.resize(
            (self.resolution + 1) * (self.resolution + 1) * (self.resolution + 1), -1)

        # Create initial voxels
        self.voxels.reserve(resolution_0 * resolution_0 * resolution_0)
        
        cdef Voxel voxel
        cdef GridPoint point
        cdef Vector3D loc
        cdef int i, j, k
        for i in range(resolution_0):
            for j in range(resolution_0): 
                for  k in range (resolution_0):
                    loc = Vector3D(
                        i * self.voxel_size_0, 
                        j * self.voxel_size_0,
                        k * self.voxel_size_0,
                    )
                    voxel = Voxel(
                        loc=loc,
                        level=0,
                        is_leaf=True,
                        positive=False,
                        negative=False,
                        queued=False,
                    )

                    assert(self.voxels.size() == vec_to_idx(Vector3D(i, j, k), resolution_0))
                    self.voxels.push_back(voxel)
        
        # Create initial grid points
        self.grid_points.reserve((resolution_0 + 1) * (resolution_0 + 1) * (resolution_0 + 1))
        for i in range(resolution_0 + 1):
            for j in range(resolution_0 + 1):
                for k in range(resolution_0 + 1):
                    loc = Vector3D(
                        i * self.voxel_size_0, 
                        j * self.voxel_size_0,
                        k * self.voxel_size_0,
                    )
                    assert(self.grid_points.size() == vec_to_idx(Vector3D(i, j, k), resolution_0 + 1))
                    self.add_grid_point(loc)

    def update(self, long[:, :] points, double[:] values):
        """Update points and set their values. Also determine all active voxels and subdivide them."""
        assert(points.shape[0] == values.shape[0])
        assert(points.shape[1] == 3)
        cdef Vector3D loc, adj_loc
        cdef long idx
        cdef int i, j, k, l
        cdef double t0 = time.perf_counter()

        # Find all indices of point and set value
        for i in range(points.shape[0]):
            loc = Vector3D(points[i, 0], points[i, 1], points[i, 2])
            idx = self.get_grid_point_idx(loc)
            if idx == -1:
                raise ValueError('Point not in grid!')
            self.grid_points[idx].value = values[i]
            self.grid_points[idx].known = True
            for j in range(-1, 1):
                for k in range(-1, 1):
                    for l in range(-1, 1):
                        adj_loc = Vector3D(loc.x + j, loc.y + k, loc.z + l)
                        self.mark_voxel(self.get_voxel_idx(adj_loc), values[i])
        self.time_set_values += time.perf_counter() - t0
        # Subdivide activate voxels and add new points
        t0 = time.perf_counter()
        self.subdivide_voxels()
        self.time_subdivide += time.perf_counter() - t0

    def query(self):
        """Query points to evaluate."""
        # Find all points with unknown value
        cdef vector[Vector3D] points
        cdef int n_unknown = 0
        for p in self.grid_points:
            if not p.known:
                n_unknown += 1 

        points.reserve(n_unknown)
        for p in self.grid_points:
            if not p.known:
                points.push_back(p.loc)

        # Convert to numpy
        points_np = np.zeros((points.size(), 3), dtype=np.int64)
        cdef long[:, :] points_view = points_np
        for i in range(points.size()):
            points_view[i, 0] = points[i].x
            points_view[i, 1] = points[i].y
            points_view[i, 2] = points[i].z

        return points_np

    def to_dense(self):
        """Output dense matrix at highest resolution."""
        cdef double t0 = time.perf_counter()
        out_array = np.full((self.resolution + 1,) * 3, np.nan)
        self.time_dense_alloc += time.perf_counter() - t0
        cdef double[:, :, :] out_view = out_array
        cdef GridPoint point
        cdef int i, j, k
        
        t0 = time.perf_counter()
        for point in self.grid_points:
            # Take voxel for which points is upper left corner
            # assert(point.known)
            out_view[point.loc.x, point.loc.y, point.loc.z] = point.value
        self.time_dense_write += time.perf_counter() - t0

        # Complete along x axis
        t0 = time.perf_counter()
        for i in range(1, self.resolution + 1):
            for j in range(self.resolution + 1):
                for k in range(self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i-1, j, k]

        # Complete along y axis
        for i in range(self.resolution + 1):
            for j in range(1, self.resolution + 1):
                for k in range(self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i, j-1, k]


        # Complete along z axis
        for i in range(self.resolution + 1):
            for j in range(self.resolution + 1):
                for k in range(1, self.resolution + 1):
                    if isnan(out_view[i, j, k]):
                        out_view[i, j, k] = out_view[i, j, k-1]
                    assert(not isnan(out_view[i, j, k]))
        self.time_dense_fill += time.perf_counter() - t0
        return out_array

    def get_profile(self):
        result = {
            'time (mise assign values)': self.time_set_values,
            'time (mise subdivide)': self.time_subdivide,
            'time (mise dense alloc)': self.time_dense_alloc,
            'time (mise dense write)': self.time_dense_write,
            'time (mise dense fill)': self.time_dense_fill,
            'mise grid points': self.grid_points.size(),
            'mise voxels': self.voxels.size(),
            'mise subdivisions': self.n_subdivided,
            'mise dense mb': (self.resolution + 1) ** 3 * 8 / 1e6,
        }
        for level in range(1, self.depth + 1):
            result['mise subdivisions level %d' % level] = self.subdivisions_by_level[level]
        return result

    def get_points(self):
        points_np = np.zeros((self.grid_points.size(), 3), dtype=np.int64)
        values_np = np.zeros((self.grid_points.size()), dtype=np.float64)

        cdef long[:, :] points_view = points_np
        cdef double[:] values_view = values_np
        cdef Vector3D loc
        cdef int i

        for i in range(self.grid_points.size()):
            loc = self.grid_points[i].loc
            points_view[i, 0] = loc.x
            points_view[i, 1] = loc.y
            points_view[i, 2] = loc.z
            values_view[i] = self.grid_points[i].value

        return points_np, values_np

    cdef void subdivide_voxels(self) except +:
        cdef vector[long] candidates
        cdef long idx

        self.active_voxels.swap(candidates)
        sort(candidates.begin(), candidates.end())
        self.voxels.reserve(self.voxels.size() + 8 * candidates.size())
        self.grid_points.reserve(self.voxels.size() + 19 * candidates.size())
        for idx in candidates:
            self.voxels[idx].queued = False
            if (self.voxels[idx].is_leaf and self.voxels[idx].level < self.depth
                    and self.voxels[idx].positive and self.voxels[idx].negative):
                self.subdivide_voxel(idx)

    cdef void subdivide_voxel(self, long idx):
        cdef Voxel voxel
        cdef GridPoint point
        cdef Vector3D loc0 = self.voxels[idx].loc
        cdef Vector3D loc
        cdef int new_level = self.voxels[idx].level + 1
        cdef int new_size = 1 << (self.depth - new_level)
        self.n_subdivided += 1
        self.subdivisions_by_level[new_level] += 1
        assert(new_level <= self.depth)
        assert(1 <= new_size <= self.voxel_size_0)

        # Current voxel is not leaf anymore
        self.voxels[idx].is_leaf = False
        # Add new voxels        
        cdef int i, j, k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    loc = Vector3D(
                        x=loc0.x + i * new_size,
                        y=loc0.y + j * new_size,
                        z=loc0.z + k * new_size,
                    )
                    voxel = Voxel(
                        loc=loc, 
                        level=new_level,
                        is_leaf=True,
                        positive=False,
                        negative=False,
                        queued=False,
                    )

                    self.voxels[idx].children[i][j][k] = self.voxels.size()
                    self.voxels.push_back(voxel)
                    if new_level < self.depth:
                        self.initialize_voxel(self.voxels.size() - 1, new_size)

        # Add new grid points
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    loc = Vector3D(
                        loc0.x + i * new_size,
                        loc0.y + j * new_size,
                        loc0.z + k * new_size,
                    )

                    # Only add new grid points
                    if self.get_grid_point_idx(loc) == -1:
                        self.add_grid_point(loc)


    @cython.cdivision(True) 
    cdef long get_voxel_idx(self, Vector3D loc) except +:
        """Utility function for getting voxel index corresponding to 3D coordinates."""
        # Shorthands
        cdef long resolution = self.resolution
        cdef long resolution_0 = self.resolution_0
        cdef long depth = self.depth
        cdef long voxel_size_0 = self.voxel_size_0

        # Return -1 if point lies outside bounds
        if not (0 <= loc.x < resolution and 0<= loc.y < resolution and 0 <= loc.z < resolution):
            return -1
        
        # Coordinates in coarse voxel grid
        cdef Vector3D loc0 = Vector3D(
            x=loc.x >> depth,
            y=loc.y >> depth,
            z=loc.z >> depth,
        )       

        # Initial voxels
        cdef int idx = vec_to_idx(loc0, resolution_0)
        cdef Voxel voxel = self.voxels[idx]
        assert(voxel.loc.x == loc0.x * voxel_size_0)
        assert(voxel.loc.y == loc0.y * voxel_size_0)
        assert(voxel.loc.z == loc0.z * voxel_size_0)

        # Relative coordinates
        cdef Vector3D loc_rel = Vector3D(
            x=loc.x - (loc0.x << depth),
            y=loc.y - (loc0.y << depth),
            z=loc.z - (loc0.z << depth),
        ) 

        cdef Vector3D loc_offset
        cdef long voxel_size = voxel_size_0

        while not voxel.is_leaf:
            voxel_size = voxel_size >> 1
            assert(voxel_size >= 1)

            # Determine child
            loc_offset = Vector3D(
                x=1 if (loc_rel.x >= voxel_size) else 0,
                y=1 if (loc_rel.y >= voxel_size) else 0,
                z=1 if (loc_rel.z >= voxel_size) else 0,
            )
            # New voxel
            idx = voxel.children[loc_offset.x][loc_offset.y][loc_offset.z]
            voxel = self.voxels[idx]

            # New relative coordinates
            loc_rel = Vector3D(
                x=loc_rel.x - loc_offset.x * voxel_size,
                y=loc_rel.y - loc_offset.y * voxel_size,
                z=loc_rel.z - loc_offset.z * voxel_size,
            ) 

            assert(0<= loc_rel.x < voxel_size)
            assert(0<= loc_rel.y < voxel_size)
            assert(0<= loc_rel.z < voxel_size)


        # Return idx
        return idx


    cdef inline void add_grid_point(self, Vector3D loc):
        cdef GridPoint point = GridPoint(
            loc=loc,
            value=0.,
            known=False,
        )
        self.grid_point_hash[vec_to_idx(loc, self.resolution + 1)] = <int>self.grid_points.size()
        self.grid_points.push_back(point)

    cdef inline int get_grid_point_idx(self, Vector3D loc):
        cdef int idx = self.grid_point_hash[vec_to_idx(loc, self.resolution + 1)]
        if idx == -1:
            return -1
        assert(self.grid_points[idx].loc.x == loc.x)
        assert(self.grid_points[idx].loc.y == loc.y)
        assert(self.grid_points[idx].loc.z == loc.z)

        return idx

    cdef inline void mark_voxel(self, long idx, double value):
        if idx != -1 and self.voxels[idx].is_leaf and self.voxels[idx].level < self.depth:
            self.voxels[idx].positive |= value >= self.threshold
            self.voxels[idx].negative |= value <= self.threshold
            if self.voxels[idx].positive and self.voxels[idx].negative and not self.voxels[idx].queued:
                self.voxels[idx].queued = True
                self.active_voxels.push_back(idx)

    cdef void initialize_voxel(self, long idx, int voxel_size):
        cdef int i, j, k, point_idx
        cdef Vector3D loc0 = self.voxels[idx].loc
        cdef Vector3D loc
        for i in range(voxel_size + 1):
            for j in range(voxel_size + 1):
                for k in range(voxel_size + 1):
                    loc = Vector3D(loc0.x + i, loc0.y + j, loc0.z + k)
                    point_idx = self.get_grid_point_idx(loc)
                    if point_idx != -1 and self.grid_points[point_idx].known:
                        self.mark_voxel(idx, self.grid_points[point_idx].value)
