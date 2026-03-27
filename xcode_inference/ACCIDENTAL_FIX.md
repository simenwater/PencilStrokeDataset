# 升降号识别修复

## 问题
组件模型只有 `accidental` 一个类，无法区分升号/降号/还原号。
iPad 端用几何规则和笔画数区分都不准，全部映射成降号。

## 解决方案
把 8 类组件扩展为 10 类：

```
旧 8 类:
0: notehead_filled
1: notehead_hollow
2: stem
3: flag
4: dot
5: rest
6: clef
7: accidental       ← 不分升降还原

新 10 类:
0: notehead_filled
1: notehead_hollow
2: stem
3: flag
4: dot
5: rest
6: clef
7: sharp            ← 升号（独立）
8: flat             ← 降号（独立）
9: natural          ← 还原号（独立）
```

## 训练数据拆分

iPad 采集数据里有 sharp/flat/natural 的原始标签。
拆分逻辑（train_component_rnn.py 的 LABEL_TO_COMPONENTS）:

```python
# 旧
'sharp':   [(0, 'accidental')],
'flat':    [(0, 'accidental')],
'natural': [(0, 'accidental')],

# 新
'sharp':   [(0, 'sharp')],
'flat':    [(0, 'flat')],
'natural': [(0, 'natural')],
```

## 执行
1. 修改 train_component_rnn.py：COMPONENT_CLASSES 改为 10 类
2. 修改 LABEL_TO_COMPONENTS：sharp/flat/natural 各自映射
3. 重新训练
4. 导出 component_best.pt，push 到 GitHub
