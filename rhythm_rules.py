"""
rhythm_rules.py
基于几何规则的节奏型识别 - 不需要训练！

原理:
- beam: 接近水平的线 (角度 < 35° 或 > 145°)
- stem: 接近垂直的线 (角度 70°-110°)
- 根据 stem 数量 + beam 数量 + 连接关系判断节奏型
"""

import json
import math
import numpy as np

def get_line_angle(points):
    """
    计算笔画的主方向角度 (0-180°)
    0° = 水平向右, 90° = 垂直向下
    """
    if len(points) < 2:
        return None

    # 用起点和终点计算主方向
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[-1][0], points[-1][1]

    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1 and abs(dy) < 1:
        return None  # 太短

    # atan2 返回 -π 到 π，转换到 0-180°
    angle = math.atan2(dy, dx) * 180 / math.pi
    if angle < 0:
        angle += 180

    return angle


def classify_stroke(points, angle_threshold_beam=35, angle_threshold_stem=20):
    """
    分类单个笔画
    返回: 'beam', 'stem', 或 'unknown'
    """
    angle = get_line_angle(points)
    if angle is None:
        return 'unknown', None

    # beam: 接近水平 (0° 或 180° 附近)
    if angle < angle_threshold_beam or angle > (180 - angle_threshold_beam):
        return 'beam', angle

    # stem: 接近垂直 (90° 附近)
    if (90 - angle_threshold_stem) < angle < (90 + angle_threshold_stem):
        return 'stem', angle

    return 'unknown', angle


def get_stroke_bbox(points):
    """获取笔画边界框"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def get_stroke_center_x(points):
    """获取笔画中心 x 坐标"""
    xs = [p[0] for p in points]
    return (min(xs) + max(xs)) / 2


def analyze_rhythm(strokes):
    """
    分析节奏型
    输入: strokes - 笔画列表，每个笔画是点列表 [[x,y], [x,y], ...]
    输出: (节奏型名称, 分析详情)
    """
    stems = []
    beams = []
    unknowns = []

    for i, stroke in enumerate(strokes):
        stroke_type, angle = classify_stroke(stroke)
        center_x = get_stroke_center_x(stroke)
        bbox = get_stroke_bbox(stroke)

        info = {
            'index': i,
            'type': stroke_type,
            'angle': angle,
            'center_x': center_x,
            'bbox': bbox,
            'points': stroke
        }

        if stroke_type == 'beam':
            beams.append(info)
        elif stroke_type == 'stem':
            stems.append(info)
        else:
            unknowns.append(info)

    # 按 x 坐标排序 stems
    stems.sort(key=lambda s: s['center_x'])

    num_stems = len(stems)
    num_beams = len(beams)

    details = {
        'num_stems': num_stems,
        'num_beams': num_beams,
        'num_unknowns': len(unknowns),
        'stems': [(s['index'], f"{s['angle']:.1f}°") for s in stems],
        'beams': [(b['index'], f"{b['angle']:.1f}°") for b in beams],
    }

    # 判断节奏型
    pattern = identify_pattern(stems, beams)

    return pattern, details


def identify_pattern(stems, beams):
    """根据 stem 和 beam 数量识别节奏型"""
    num_stems = len(stems)
    num_beams = len(beams)

    # 2 stems
    if num_stems == 2:
        if num_beams == 1:
            return 'beam_8_8'      # 两个八分
        elif num_beams == 2:
            return 'beam_16_16'   # 两个十六分

    # 3 stems
    elif num_stems == 3:
        if num_beams == 1:
            return 'beam_triplet'  # 三连音
        elif num_beams == 2:
            # 需要判断是 16_16_8 还是 8_16_16
            # 看哪条 beam 更短（次 beam）
            return identify_3_stem_pattern(stems, beams)

    # 4 stems
    elif num_stems == 4:
        if num_beams == 2:
            return 'beam_16x4'    # 四个十六分

    return f'unknown_{num_stems}s_{num_beams}b'


def identify_3_stem_pattern(stems, beams):
    """
    区分 beam_16_16_8 和 beam_8_16_16
    看次 beam 连接的是前两个还是后两个 stem
    """
    if len(beams) != 2:
        return 'unknown_3stem'

    # 找出两条 beam 的长度
    beam_lengths = []
    for b in beams:
        bbox = b['bbox']
        width = bbox[2] - bbox[0]  # max_x - min_x
        beam_lengths.append((width, b))

    beam_lengths.sort(key=lambda x: x[0])
    short_beam = beam_lengths[0][1]  # 次 beam (较短)

    # 次 beam 的中心 x
    short_beam_cx = (short_beam['bbox'][0] + short_beam['bbox'][2]) / 2

    # stems 的中心 x
    stem_cxs = [s['center_x'] for s in stems]
    mid_x = (stem_cxs[0] + stem_cxs[2]) / 2  # 第一个和第三个 stem 的中点

    if short_beam_cx < mid_x:
        return 'beam_16_16_8'  # 次 beam 在左边 → 前十六后八
    else:
        return 'beam_8_16_16'  # 次 beam 在右边 → 前八后十六


def test_with_file(data_path='collected/iPad.jsonl'):
    """用现有数据测试规则"""
    print("=" * 60)
    print("几何规则节奏型识别测试")
    print("=" * 60)

    # 测试已有的 beam 数据 (如果有的话)
    results = {}

    with open(data_path) as f:
        for line in f:
            sample = json.loads(line)
            label = sample['label']
            strokes = sample['strokes']

            # 只测试 beam 相关标签
            if not label.startswith('beam_'):
                continue

            # 转换格式
            stroke_points = []
            for stroke in strokes:
                points = [[p[0], p[1]] for p in stroke]
                stroke_points.append(points)

            pattern, details = analyze_rhythm(stroke_points)

            if label not in results:
                results[label] = {'correct': 0, 'wrong': 0, 'predictions': []}

            is_correct = (pattern == label)
            if is_correct:
                results[label]['correct'] += 1
            else:
                results[label]['wrong'] += 1
                results[label]['predictions'].append(pattern)

    if not results:
        print("\n没有找到 beam 数据，先采集一些测试！")
        print("\n手动测试模式:")
        test_manual()
        return

    print("\n测试结果:")
    for label, stats in results.items():
        total = stats['correct'] + stats['wrong']
        acc = 100 * stats['correct'] / total if total > 0 else 0
        print(f"  {label}: {acc:.1f}% ({stats['correct']}/{total})")
        if stats['wrong'] > 0:
            print(f"    错误预测: {stats['predictions'][:5]}")


def test_manual():
    """手动测试几个例子"""
    print("\n" + "=" * 40)
    print("模拟测试")
    print("=" * 40)

    # 模拟 beam_8_8: 2 个竖线 + 1 个横线
    test_cases = [
        {
            'name': 'beam_8_8 (2竖+1横)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1 (垂直)
                [[30, 10], [30, 50]],   # stem 2 (垂直)
                [[10, 10], [30, 12]],   # beam (略斜的横线)
            ]
        },
        {
            'name': 'beam_16_16 (2竖+2横)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1
                [[30, 10], [30, 50]],   # stem 2
                [[10, 10], [30, 10]],   # beam 1
                [[10, 15], [30, 15]],   # beam 2
            ]
        },
        {
            'name': 'beam_16x4 (4竖+2横)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1
                [[20, 10], [20, 50]],   # stem 2
                [[30, 10], [30, 50]],   # stem 3
                [[40, 10], [40, 50]],   # stem 4
                [[10, 10], [40, 12]],   # beam 1
                [[10, 15], [40, 17]],   # beam 2
            ]
        },
        {
            'name': 'beam_triplet (3竖+1横)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1
                [[25, 10], [25, 50]],   # stem 2
                [[40, 10], [40, 50]],   # stem 3
                [[10, 10], [40, 10]],   # beam
            ]
        },
        {
            'name': 'beam_16_16_8 (3竖+2横，短横在左)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1
                [[25, 10], [25, 50]],   # stem 2
                [[40, 10], [40, 50]],   # stem 3
                [[10, 10], [40, 10]],   # 主 beam (长)
                [[10, 15], [25, 15]],   # 次 beam (短，在左边)
            ]
        },
        {
            'name': 'beam_8_16_16 (3竖+2横，短横在右)',
            'strokes': [
                [[10, 10], [10, 50]],   # stem 1
                [[25, 10], [25, 50]],   # stem 2
                [[40, 10], [40, 50]],   # stem 3
                [[10, 10], [40, 10]],   # 主 beam (长)
                [[25, 15], [40, 15]],   # 次 beam (短，在右边)
            ]
        },
        {
            'name': '斜beam测试 (beam倾斜25°)',
            'strokes': [
                [[10, 10], [12, 50]],   # stem 1 (略斜)
                [[30, 0], [32, 40]],    # stem 2 (略斜)
                [[10, 10], [30, 0]],    # beam (明显斜)
            ]
        },
    ]

    for tc in test_cases:
        pattern, details = analyze_rhythm(tc['strokes'])
        print(f"\n{tc['name']}")
        print(f"  识别结果: {pattern}")
        print(f"  详情: {details['num_stems']} stems, {details['num_beams']} beams")


if __name__ == '__main__':
    test_with_file()
