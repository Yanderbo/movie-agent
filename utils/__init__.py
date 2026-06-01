# -*- coding: utf-8 -*-
"""通用工具函数。"""
from typing import Callable, TypeVar

T = TypeVar("T")


def group_consecutive(items: list[T], index_of: Callable[[T], int]) -> list[list[T]]:
    """
    将一组对象按其整数序号是否连续切分为若干分组。

    用于理解流水线的层级覆盖兜底：把"未被分配"的 shot / beat / story_scene
    按相邻关系聚合为过渡单元，避免散落丢失。

    Args:
        items: 待分组对象列表。
        index_of: 从对象取出整数序号的函数（如 shot -> scene_index）。

    Returns:
        分组列表，每组内的序号严格连续（step=1），组按序号升序排列。
    """
    groups: list[list[T]] = []
    current: list[T] = []
    for it in sorted(items, key=index_of):
        if current and index_of(it) != index_of(current[-1]) + 1:
            groups.append(current)
            current = []
        current.append(it)
    if current:
        groups.append(current)
    return groups
