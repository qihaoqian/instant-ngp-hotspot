## experienment on hotspot

### Experiment 1

using  

+ boundary_loss_weight: 350 
+ eikonal_loss_weight: 3 
+ sign_loss_weight: 0  
+ heat_loss_weight: 16
+ original heat loss code. main difference is (loss_heat /= X_free.reshape(-1, 2).shape[0] # average over batch size)
+ <img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416164708091.png" alt
  ="image-20250416164708091" style="zoom:33%;" />

The heat loss is very large. And not stable.

---

### Experiment 2

Try to lower the heat loss weight to 1, from 16 , loss curve look better but result not good

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416185351878.png" alt="image-20250416185351878" style="zoom: 33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416185545718.png" alt="image-20250416185545718" style="zoom:33%;" />

---

### Experiment 3

use larger boundary loss and lower heat loss to 0.01

+ boundary_loss_weight: 1000
+ eikonal_loss_weight: 3

+ sign_loss_weight: 0

+ heat_loss_weight: 0.01

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416190845599.png" alt="image-20250416190845599" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416190813977.png" alt="image-20250416190813977" style="zoom:33%;" />

Result looks better, boundary loss is converge good , but still not correct at border.

---

### Experiment 4

continue lower heat loss to 0.001, plan to look into origin code loss weight. Maybe need change the weight during training.

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416192124544.png" alt="image-20250416192124544" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416192146195.png" alt="image-20250416192146195" style="zoom:33%;" />

This object is not looking good and noise at border.

---

### Original code loss

boundary: 0.0977 *350 = 34

eikonal :0.85 *1.0 = 0.85

heat: 7.75 *20 = 155

GPT说 lambda 越大，PDE 惩罚对远离表面的区域越敏感，可以调大试试边缘会不会有噪声。

```
[loss]
loss_type = "igr_w_heat"
loss_weights = [350, 0, 0, 1, 0, 0, 20]
heat_decay = "linear"
heat_decay_params = [20, 0.8, 20, 0.1]
eikonal_decay = "linear"
eikonal_decay_params = [1, 0.2, 1, 10]
heat_lambda_decay = "linear"
heat_lambda_decay_params = [4, 0.2, 4, 30]
```

heat and eikonal and heat lambda have decay mechanism

---

### Experiment 5

change loss weight according to original code , Now we can see inner object has sdfs < 0. But still noise at border. 
I think this is a problem with loss weight and heat lambda decay mechanism in the original code.  

GPT says that the larger the lambda, the more sensitive the PDE penalty is to areas far from the surface.

original lam is 4 and have decay mechanism = [4, 0.2, 4, 30]

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416194803028.png" alt="image-20250416194803028" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416194743453.png" alt="image-20250416194743453" style="zoom:33%;" />

---

### Experiment 6

Try lambda = 20 output.ply has 1440000 faces, when lambda=8 output.ply has 3280000faces, larger lambda has better performance. Continue enlarge it.

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416211001232.png" alt="image-20250416211001232" style="zoom:33%;" />

---

**Should Try to look loss curve of original code**

---

### Experiment 7

-   boundary_loss_weight: 1000
-   eikonal_loss_weight: 0.1
-   sign_loss_weight: 0
-   heat_loss_weight: 0.001
-   heat_loss_lambda: 60

output.ply has 2047410 faces. Not good , now try lam = 120

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416212114178.png" alt="image-20250416212114178" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416212130783.png" alt="image-20250416212130783" style="zoom:33%;" />

---

### Experiment 8

-   boundary_loss_weight: 1000
-   eikonal_loss_weight: 0.1
-   sign_loss_weight: 0
-   heat_loss_weight: 0.001
-   heat_loss_lambda: 120

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416213303423.png" alt="image-20250416213303423" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250416213342326.png" alt="image-20250416213342326" style="zoom:33%;" />

As lambda grow, noise in border become less

---

Yesterday got a big mistake. Redo all the experiment to find best heat loss lambda. Try warm up mechanism.

---

### Lambda Increase by epoch

Try **lam = epoch * 5.**

-   boundary_loss_weight: 1000
-   eikonal_loss_weight: 0.1
-   sign_loss_weight: 0
-   heat_loss_weight: 5
-   heat_loss_lambda: 5
-   epoch = 25

For the mesh result. Noise at border become less as lambda increase with epoch.

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163530759.png" alt="image-20250417163530759" style="zoom: 50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163553568.png" alt="image-20250417163553568" style="zoom: 50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163829588.png" alt="image-20250417163829588" style="zoom:25%;" /><img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163842424.png" alt="image-20250417163842424" style="zoom:25%;" /><img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163900875.png" alt="image-20250417163900875" style="zoom:25%;" /><img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163918223.png" alt="image-20250417163918223" style="zoom:25%;" />



<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163936764.png" alt="image-20250417163936764" style="zoom:25%;" />





---



### Lambda = 200

-   boundary_loss_weight: 1000
-   eikonal_loss_weight: 0.1
-   sign_loss_weight: 0
-   heat_loss_weight: 8
-   heat_loss_lambda: 200

converge good. Result good. But the section not correct. sdfs of inner object > 0.

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163307205.png" alt="image-20250417163307205" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417163326806.png" alt="image-20250417163326806" style="zoom:33%;" />

---

### Lambda = 100

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417164956330.png" alt="image-20250417164956330" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417165113058.png" alt="image-20250417165113058" style="zoom:33%;" />

---

### Lambda = 50

only a little border noise. Sdfs of inner object > 0

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417165711363.png" alt="image-20250417165711363" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417165724314.png" alt="image-20250417165724314" style="zoom:33%;" />

---

### Lambda = 20

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417172052370.png" alt="image-20250417172052370" style="zoom: 67%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417172406565.png" alt="image-20250417172406565" style="zoom: 67%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417172125511.png" alt="image-20250417172125511" style="zoom: 50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417172203233.png" alt="image-20250417172203233" style="zoom:33%;" />

---

### Lambda = 10

More border noise as lambda decrease.

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417173214626.png" alt="image-20250417173214626" style="zoom:67%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417173325630.png" alt="image-20250417173325630" style="zoom:67%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417173340829.png" alt="image-20250417173340829" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417173502381.png" alt="image-20250417173502381" style="zoom:33%;" />

---

### Lambda Decay

lam = 100  -- epoch <= 15

lam = 20    -- epoch > 15

epoch = 25

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417181602834.png" alt="image-20250417181602834" style="zoom: 50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417181648815.png" alt="image-20250417181648815" style="zoom:67%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417181711514.png" alt="image-20250417181711514" style="zoom: 33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250417181756223.png" alt="image-20250417181756223" style="zoom:33%;" />

---

**What we know so far:**

1. When the heat loss lambda is large (>50), the mesh boundaries are clean without noise. However, the interior of the object shows SDF values greater than 0.
2. When the heat loss lambda is small (≤20), the mesh boundaries become noisy, and the noise increases as lambda decreases. The interior of the object shows SDF values less than 0.
3. The original Hotspot code can correctly predict the SDF cross-section for our dataset.
4. The sdf value cross section 

------

**Possible causes of this issue:**

1. The heat loss might not work well with NGP.
   - Try replacing the hash encoding with a traditional one and test again.
2. We may need to apply loss weight decay and lambda decay like in the original Hotspot implementation.
   - The boundary, eikonal, and heat losses can still converge.
   - Try using the Hotspot decay mechanism.
3. There might be a bug in our codebase.
   - However, using MAPE loss gave correct predictions by only changing the loss function in the training step.

---

### 4.25

生成的sdf value全部>0 所以无法生成mesh， because threshold = 0.

after replace the network(), the issue with border noise disappear. But the inner object sdf value > 0, I think I can solve this by try different lambda. This is result of lambda = 20.

可见 `u` 的所有值都落在 **[0.062, 0.847]** 之间，**并不包含 0**。而 Marching Cubes 只有在场值跨越阈值（threshold）时才会生成等值面。你把 `threshold=0`，但 `u` 全都比 0 大，根本没发生“0”交叉，自然输出空的 `(0,3)`。

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250425220449089.png" alt="image-20250425220449089" style="zoom: 25%;" />

+ 尝试更小的lambda
+ dataset准备时候不增加扰动的点，只使用表面点和free点

---

### QuaNet Lam=5

-   boundary_loss_weight: 1000
-   eikonal_loss_weight: 1
-   sign_loss_weight: 0
-   heat_loss_weight: 100
-   heat_loss_lambda: 5

手指细节不清晰，背上的mesh有洞，sdf小于0的部分少，物体内部sdf值没有按距离递减

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502003627997.png" alt="image-20250502003627997" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502003351018.png" alt="image-20250502003351018" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502003650711.png" alt="image-20250502003650711" style="zoom:67%;" />

---

### QuaNet Lam=10

-   boundary_loss_weight: 100
-   eikonal_loss_weight: 1
-   sign_loss_weight: 0
-   heat_loss_weight: 100
-   heat_loss_lambda: 10

物体内部的sdf值较小，手指细节仍然存在问题

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502004006798.png" alt="image-20250502004006798" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502004908369.png" alt="image-20250502004908369" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250502004929441.png" alt="image-20250502004929441" style="zoom:80%;" />

---

using the same network as original code, why not work?

**problem:**

+ not accurate at finger detail
+ hole on the mesh
+ sdf value not correct inside

**why?**

1. 采样点数量过多，总共才49990个点，我采样了65000多个，原代码采样了15000个surface和15000个free
2. 原代码对于free point的采样方法是uniform和central_gaussian混合并计算了pdf分布
3. 我的缩放方法和原代码不一致，我是在mesh上操作，它是在point上操作



**原代码采取固定权重和lambda计算：**

- boundary_loss_weight: 350
- eikonal_loss_weight: 1
- heat_loss_weight: 20
- heat_loss_lambda: 4

我也如此计算了



---

### QuaNet lam = 4  

boundary_loss_weight: 350

eikonal_loss_weight: 1

heat_loss_weight: 20

heat_loss_lambda: 4

lr: 5e-5

<img src="E:\Downloads\imageData (1).png" alt="imageData (1)" style="zoom: 25%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250507145906131.png" alt="image-20250507145906131" style="zoom:50%;" />

原code loss
<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250507162323902.png" alt="image-20250507162323902" style="zoom: 50%;" /><img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250507162353352.png" alt="image-20250507162353352" style="zoom:50%;" /><img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250507162407713.png" alt="image-20250507162407713" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250507145951336.png" alt="image-20250507145951336" style="zoom:50%;" />

+ 裁剪
+ sign+ eikonal 对比 hotspot
+ 对比 sign+hotspot （收敛速度，达到相同error所需iteration；对比sdf error，对比mesh error）

---

in encoder data map to [0,1]

原代码的eikonal和heat loss收敛到了0.02和0.07，我只能做到0.26和0.3. 导致了背上的破洞

+ 原代码使用torch autograd
+ 网络不同
+ save时候做了裁剪（不影响网络结果）
+ 求梯度的eps过大？
+ grid编码分辨率太高？
+ dataset free点的比例太高! yes

---

### hotspot_lam_4_2

-   boundary_loss_weight: 350
-   eikonal_loss_weight: 1
-   sign_loss_weight: 0
-   heat_loss_weight: 40
-   h: 1e-3
-   heat_loss_lambda: 4
-   lr: 1e-4
-   num_samples: 100000

20epoch就可以了

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250509185335665.png" alt="image-20250509185335665" style="zoom: 33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250509185604858.png" alt="image-20250509185604858" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250509185646156.png" alt="image-20250509185646156" style="zoom:33%;" />

---

更改N_max, N_min导致距离远的地方sdf值为负。

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250509203245922.png" alt="image-20250509203245922" style="zoom:50%;" />

---

N_min can not be change must be 16

increase or decrease the N_max does not change the result, the max sdf value is still 0.25.

Because when it get to the border area, the sdf value will slowly decrease to 0

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250510110900510.png" alt="image-20250510110900510" style="zoom:33%;" />

---

  boundary_loss_weight: 350

  eikonal_loss_weight: 1

  sign_loss_weight: 0

  heat_loss_weight: 20

  heat_loss_lambda: 4

  num_layers: 4

  hidden_dim: 128

  num_levels: 8

  base_resolution: 16

  desired_resolution: 2048

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511115438892.png" alt="image-20250511115438892" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511115455899.png" alt="image-20250511115455899" style="zoom:50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511115548850.png" alt="image-20250511115548850" style="zoom:33%;" />

边缘出现0

---

  num_layers: 6

  hidden_dim: 256

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511120630840.png" alt="image-20250511120630840" style="zoom: 50%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511120650066.png" alt="image-20250511120650066" style="zoom:33%;" />

<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250511120710344.png" alt="image-20250511120710344" style="zoom: 50%;" />

---

加入了球面后处理和quanet中的init。         

sphere_radius=1.6,

sphere_scale=1.0,

```python
        # 球面后处理：符号+开方 -> 减半径 -> 缩放
        eps = 1e-8
        h_sqrt     = torch.sign(h) * torch.sqrt(h.abs() + eps)
        h_centered = h_sqrt - self.sphere_radius
        h = h_centered * self.sphere_scale
```

heat loss降低到了0.1

但是符号方向反了



<img src="C:\Users\23638\AppData\Roaming\Typora\typora-user-images\image-20250512132210845.png" alt="image-20250512132210845" style="zoom:50%;" />

eikonal和heat loss都无法限制得到正确的sdf value符号
