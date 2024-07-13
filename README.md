## mamba-minimal

在一个 PyTorch 文件中简单、最小化地实现 Mamba。

特点
* 正向和反向传递的数值输出与官方实现相同
* 简化、可读、带注释的代码

不包括
* 速度。官方实现进行了大量优化，这些优化是 Mamba 论文的核心贡献。为了提高可读性，我对大部分实现进行了简化。
* 适当的参数初始化（当然也可以在不影响可读性的前提下进行添加）

## Demo

See [demo.ipynb](demo.ipynb) for examples of prompt completions.

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba is the world's longest venomous snake with an estimated length of over 150 m. With such a large size and a venomous bite, Mamba kills by stabbing the victim (which is more painful and less effective than a single stab of the bite)
> 
测试[mamba_minmal.ipynb](mamba_minmal.ipynb)demo可以直接在colab上面运行

## References

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
