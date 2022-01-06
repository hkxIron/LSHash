# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

# wiki:https://yongyuan.name/blog/ann-search.html

import os
import json
import numpy as np

from storage import storage

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


class LSHash(object):
    """ LSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict` or `redis`, and `config` is the
        configuration used by the backend. For `redis` it should be in the
        format of `{"redis": {"host": hostname, "port": port_num}}`, where
        `hostname` is normally `localhost` and `port` is normally 6379.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):
        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            # 生成num_hashtable个随机划分表,每个表维度:[hash_size, dim]
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """
        # 初始化num_hashtable个hash表
        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """
        # 随机矩阵[hash_size, input_dim], 矩阵每行代表一条直线的法向量,Ax>0即代表与所有平面的法向量的夹角<90的所有点x的集合
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """

        try:
            input_point = np.array(input_point)  # for faster dot product
            # planes:[hash_size, dim]
            # input_point:[dim], 亦可理解为input_point: [dim,1]
            # projections:[hash_size] = planes*input_point
            # 矩阵planes与input_points相乘的几何意义就是判断input_points是否与planes每行的直线法向量的乘积
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSHash instance""", e)
            raise
        else:
            # 比如可能是'001'
            return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, json_or_tuple):
        """ Takes either a JSON-serialized data structure or a tuple that has
        the original input points stored, and returns the original input point
        in numpy array format.
        """
        if isinstance(json_or_tuple, str):
            # JSON-serialized in the case of Redis
            try:
                # Return the point stored as list, without the extra data
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            # in this case extra data exists
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def index(self, input_point, extra_data=None):
        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table. `extra_data` needs to be JSON serializable if in-memory
        dict is not used as storage.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
            This object will be converted to Python tuple and stored in the
            selected storage.
        :param extra_data:
            (optional) Needs to be a JSON-serializable object: list, dicts and
            basic types such as strings and integers.
        """

        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()

        if extra_data:
            value = (tuple(input_point), extra_data)
        else:
            value = tuple(input_point)

        # 每个hash表均要存一下hash串+原始point
        for i, table in enumerate(self.hash_tables):
            hash_code = self._hash(self.uniform_planes[i], input_point)
            table.append_val(key=hash_code, val=value)

    def query(self, query_point, num_results=None, distance_func_for_hash=None):
        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func_for_hash:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """

        candidates = set()
        if not distance_func_for_hash:
            distance_func_for_hash = "euclidean"

        if distance_func_for_hash == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hash_tables):
                binary_hash_for_query = self._hash(self.uniform_planes[i], query_point)
                for key in table.keys():
                    distance = LSHash.hamming_dist(key, binary_hash_for_query)
                    # 所有hamming距离<2的全都加入候选集合,注意,不一定是相同hash_key下的所有候选point
                    if distance < 2:
                        # 将该hash_key下所有的原始值全加入set
                        candidates.update(table.get_list(key))

            d_func_for_rank = LSHash.euclidean_dist_square

        else: # euclidean

            if distance_func_for_hash == "euclidean":
                d_func_for_rank = LSHash.euclidean_dist_square
            elif distance_func_for_hash == "true_euclidean":
                d_func_for_rank = LSHash.euclidean_dist
            elif distance_func_for_hash == "centred_euclidean":
                d_func_for_rank = LSHash.euclidean_dist_centred
            elif distance_func_for_hash == "cosine":
                d_func_for_rank = LSHash.cosine_dist
            elif distance_func_for_hash == "l1norm":
                d_func_for_rank = LSHash.l1norm_dist
            else:
                raise ValueError("The distance function name is invalid.")

            # 只有hash值相同的才认为是候选集合,只要有一个hash表认为是候选就加入候选
            for i, table in enumerate(self.hash_tables):
                binary_hash_for_query = self._hash(self.uniform_planes[i], query_point)
                candidates.update(table.get_list(binary_hash_for_query))

        # rank candidates by distance function
        # 计算query与每个候选集原始值的距离
        # [(candidate_point, distance),... ]
        candidates = [(candidate_point, d_func_for_rank(query_point, self._as_np_array(candidate_point)))
                      for candidate_point in candidates]
        candidates.sort(key=lambda x: x[1]) # 按距离升序排序
        # 选出距离最近的topK
        return candidates[:num_results] if num_results else candidates

    ### distance functions

    # 海明距离是直接异或么?直接数不同的位数的个数
    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        return np.sqrt(LSHash.euclidean_dist_square(x,y))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)


if __name__ == '__main__':
    lsh = LSHash(hash_size=6, input_dim=8, num_hashtables=3)
    # 给数据建立hash索引
    lsh.index(input_point=[1, 2, 3, 4, 5, 6, 7, 8])
    lsh.index(input_point=[2, 3, 4, 5, 6, 7, 8, 9])
    lsh.index(input_point=[1, 2, 3, 4, 4, 6, 7, 8])
    lsh.index(input_point=[1, 2, 3, 3, 5, 6, 7, 8])
    lsh.index(input_point=[1, 2, 3, 4, 5, 6, 7, 9])
    lsh.index(input_point=[2, 2, 3, 4, 5, 6, 7, 9])
    lsh.index(input_point=[2, -2, 3, 4, 5, 6, 7, 9])
    lsh.index(input_point=[-1, 2, 3, 4, 5, 6, 7, 9])
    lsh.index(input_point=[10, 12, 99, 1, 5, 31, 2, 3])
    # 查询
    res = lsh.query(query_point=[1, 2, 3, 4, 5, 6, 7, 7], num_results=4)
    print(res)
