# torch-ngp-hotspot

### Install with pip
```bash
pip install -r requirements.txt

### Build extension 
# install manually, here is an example:
cd gridencoder
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```



# Usage

We use the same data format as instant-ngp, e.g., [armadillo](https://github.com/NVlabs/instant-ngp/blob/master/data/sdf/armadillo.obj) and [fox](https://github.com/NVlabs/instant-ngp/tree/master/data/nerf/fox). 
Please download and put them under `./data`.

We also support self-captured dataset and converting other formats (e.g., LLFF, Tanks&Temples, Mip-NeRF 360) to the nerf-compatible format, with details in the following code block.

<details>
  <summary> Supported datasets </summary>

  * [nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 

  * [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip): [[conversion script]](./scripts/tanks2nerf.py)

  * [LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7): [[conversion script]](./scripts/llff2nerf.py)

  * [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip): [[conversion script]](./scripts/llff2nerf.py)

  * (dynamic) [D-NeRF](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)

  * (dynamic) [Hyper-NeRF](https://github.com/google/hypernerf/releases/tag/v0.1): [[conversion script]](./scripts/hyper2nerf.py)

</details>

First time running will take some time to compile the CUDA extensions.

```bash
### Instant-ngp hotspot
python main.py --config config/hotspot.yaml #--trainer.eval-interval 3 --optimizer.lr 0.0001
```


