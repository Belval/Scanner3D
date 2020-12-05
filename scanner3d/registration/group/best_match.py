"""
BestMatchReg is a custom implementation of matching across a set of clouds.
Each iteration, point clouds are joined based on their closest match (not necessarily 1-1) until only one
cloud remains.
"""

import logging
import numpy as np
import open3d as o3d

from scanner3d.registration.group.group.base_group_reg import BaseGroupReg
from scanner3d.registration.pair.filterreg_pair_reg import FilterReg


class BestMatchReg(BaseGroupReg):
    def __init__(self, pair_reg=FilterReg(), min_fit=0.97):
        self.pair_reg = pair_reg
        self.min_fit = min_fit

    def register(self, pcds):
        meta_iter = 0
        while len(pcds) > 1:
            scores = []
            for i, pcd1 in enumerate(pcds):
                for j, pcd2 in enumerate(pcds):
                    if i >= j:
                        continue
                    trans, fit = self.pair_reg.register(pcd1, pcd2)
                    scores.append((i, j, fit, trans))
            num_list = [i for i in range(len(pcds))]
            scores.sort(key=lambda x: x[2])
            new_pcds = []
            while num_list and scores:
                i, j, fit, transformation = scores.pop()

                if i not in num_list and j not in num_list:
                    continue
                if fit < self.min_fit:
                    continue

                target = copy.deepcopy(pcds[j])
                source = copy.deepcopy(pcds[i])
                if isinstance(transformation, np.ndarray):
                    source.transform(transformation)
                else:
                    source.points = transformation.transform(source.points)
                source.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                new_pcd = target + source
                new_pcd = new_pcd.voxel_down_sample(voxel_size=0.01)
                new_pcds.append(new_pcd)
                # TODO: Add project-wide debug options
                # o3d.io.write_point_cloud(f"new_clouds/{meta_iter}_{i}_{j}_{fit}.pcd", pcd)
                try:
                    num_list.remove(i)
                except:
                    pass
                try:
                    num_list.remove(j)
                except:
                    pass
            pcds = new_pcds
            meta_iter += 1
        return pcds[0]
