### 1
second_gradient_loss_weight: 1
loss 过小不下降

### 2
second_gradient_loss_weight: 5
下降到0.3，导致eikonal loss 在0.9难以下降，总体的一阶grad很小，sdf value也很小

### 3
second_gradient_loss_weight: 3
sec_grad loss到0.62， eikonal loss在0.75,两者有对抗关系。总体的一阶grad很小，sdf value也很小

### 4
second_gradient_loss_weight: 1
调整h2 from 0.1 to 0.01
Average free point gradients norm: 5.926588528382126e-06

这样的方法似乎不对，因为eikonal限制处处为1. 要从方向着手，现在的方法是带方向的，要先方向对才行，可以norm之后忽略方向

### 5 
```
dot_prod = (grads_1 * grads_2).sum(dim=-1, keepdim=True)  
sec_scalar = (grads_1.norm(dim=-1, keepdim=True) - grads_2.norm(dim=-1, keepdim=True)) / (2 * h2)   
sec_grads = torch.where(
    dot_prod < 0,                     # 条件
    grads_1 + grads_2,                # True 分支 → 向量和
    sec_scalar.expand_as(grads_1)     # False 分支 → 标量结果按通道广播
)
```
second_gradient_loss_weight: 1
h1: 0.01
h2: 0.05

loss很高，难以奏效。不谈方向没有太大意义，可能可以阻止符号的转换，因为dot_prod < 0, sec_grad = grads_1 + grads_2
