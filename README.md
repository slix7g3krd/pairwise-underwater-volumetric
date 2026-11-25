# pairwise-underwater-volumetric

## 领域

游戏、图形与多媒体

## 主题

水下体积吸收散射

## 说明

按深度衰减、视角相关折射与雾和景深顺序。

初始状态下 `graphics/underwater_volume.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
