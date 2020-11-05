# Co-Section

![Teaser image for Co-Section](images/teaser.png)

This repository provides source code for Co-Section accompanying the following publication:

*Michael Strecke and Joerg Stuecker, "**Where Does It End? - Reasoning About
Hidden Surfaces by Object Intersection Constraints**"*  
*Presented at the **IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) 2020**, (virtual conference)*

Please see the [project page](https://cosection.is.tue.mpg.de/) for details.

If you use the source code provided in this repository for your research, please cite the corresponding publication as:
```
@InProceedings{strecke2020_cosection,
  author      = {Michael Strecke and Joerg Stueckler},
  booktitle   = {2020 {IEEE}/{CVF} Conference on Computer Vision and Pattern Recognition ({CVPR})},
  title       = {Where Does It End? - Reasoning About Hidden Surfaces by Object Intersection Constraints},
  year        = {2020},
  month       = {jun},
  publisher   = {{IEEE}}
}
```

## Getting started

### 1. Setup EM-Fusion

Our code builds upon EM-Fusion and needs the same requirements. Install the
requirements listed in the [EM-Fusion readme](https://gitlab.localnet/embodied-vision/ev-projects-michael-strecke/code/emfusion-release#0-install-dependencies) and set up Mask R-CNN as described
there. The easiest way to do this is to clone this repository with 
`git clone --recursive` change to `external/emfusion` and work through steps 0
and 1 of the EM-Fusion Getting started instructions in.

If you already have a setup working for EM-Fusion, you can instruct Co-Section
to use it with the `-DEMFUSION_DIR=<path/to/emfusion>` flag. Remember to also
set `MASKRCNN_ROOT_DIR` and `MASKRCNN_VENV_DIR` if those deviate from the
EM-Fusion default for your setup.

### 2. Build Co-Section

After installing all dependencies mentioned and setting up EM-Fusion,
create a build directory in the root directory of the Co-Section code, configure the build using CMake, and build the
project:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

Depending on where you installed the dependencies (including those of EM-Fusion), you might want to add 
`-DCMAKE_PREFIX_PATH=<path/to/install>` to the `cmake` call.

### 3. Running the code

Change to the `build` folder for running the code (as for EM-Fusion, we place
the `maskrcnn.py` file there so this should be the working directory for running
the code).

The `build` folder will contain all the `preprocess_masks` and `EM-Fusion`
executables from EM-Fusion. We recommend preprocessing masks if you run into GPU
memory issues.

The main executable is called `Co-Section` and provides the following options:

```
$ ./Co-Section -h
Co-Section: Dynamic tracking and Mapping from RGB-D data with optimization of 3D volumes:

One of these options is required:
  -t [ --tumdir ] arg       Directory containing RGB-D data in the TUM format
  -d [ --dir ] arg          Directory containing color and depth images

Possibly needed when using "--dir" above:
  --colordir arg (=colour)  Subdirectory containing color images named 
                            Color*.png. Needed if different from "colour"
  --depthdir arg (=depth)   Subdirectory containing depth images named 
                            Depth*.exr. Needed if different from "depth"

Optional inputs:
  -h [ --help ]             Print this help
  -e [ --exportdir ] arg    Directory for storing results
  --background              Whether to run this program without live output 
                            (without -e it won't be possible to examine 
                            results)
  --show-slices             Whether to show ESDF slices in the 3D 
                            visualization.
  -c [ --configfile ] arg   Path to a configuration file containing experiment 
                            parameters
  -m [ --maskdir ] arg      Directory containing preprocessed Mask R-CNN 
                            results

```

Most flags as well as the visualization windows behave the same as for EM-Fusion
(see the EM-Fusion README).

There are three changes in the application interface compared to EM-Fusion:
1. 3D visualization is enabled by default for Co-Section if not run in background mode
2. There is a new flag `--show-slices`, which enables SDF slice visualization in the 3D preview window
3. The config files accept some more options for Co-Section algorithm parameters (see [data.h](include/CoSection/optim/data.h) for documentation).

#### 3.1 Rendering images

The `renderdata` executable allows for interactively exploring reconstructions.
```
$ ./renderdata -h
Render 3D models created by EM-Fusion or Co-Section:

Required input:
  -d [ --dir ] arg       EM-Fusion or Co-Section output directory

Optional flags:
  -h [ --help ]          Print this help
  -o [ --opath ] arg     Output path. Should be a folder. Files will be written
                         as opath/<framenum>.png
  -f [ --frames ] arg    Which frame(s) to render. Either single number n or 
                         range n-m
  -t [ --tsdf ]          Render TSDF models instead of optimized ones
  -p [ --pose ] arg      Pose from which to render the 3D models
  --outpose arg          Output file to write final pose to (for reproducing 
                         results)
  --followcam            Whether to follow the camera viewpoint of the 
                         recording
```
Without the `-o` flag, this program lets you navigate in the 3D scene
interactively to find good view points for then generating images. The playback
can be paused with `SPACE`, reversed with `LEFT` and continued forward with
`RIGHT`. Closing the window or pressing `Q` ends the program. The `--outpose`
flag lets you save the last viewer pose before ending the program so it can be
reused. With the `-o` flag given, the program will run non-interactively and
output all frames or the given range (with `-f`) to the output folder.

With the `-p` flag, you can load a previously stored pose to render a scene from
the same viewpoint. The `--followcam` flag will render the scene from the
estimated camera pose by EM-Fusion/Co-Section.

If a `calibration.txt` like in the `car4-full` archive of the Co-Fusion datasets
is present in the directory given by `-d`, the camera parameters for rendering
will be adapted to it.

The `-t` flag switches between optimized and TSDF meshes.

For more high-quality renderings, we provide a python script [renderdata.py](apps/renderdata.py)
to work with [blender](http://blender.org):
```
blender --background --python ../apps/renderdata.py -- -h
Blender 2.82 (sub 7) (hash 77d23b0bd76f built 2020-02-12 17:14:50)
Read prefs: /is/sg/mstrecke/.config/blender/2.82/config/userpref.blend
found bundled python: /is/sg/mstrecke/install/blender-2.82-linux64/2.82/python
usage: blender [-h] --dir DIR --opath OPATH [--frames FRAMES] [--tsdf]
               [--pose POSE]

Render 3D models created by EM-Fusion or Co-Section

optional arguments:
  -h, --help            show this help message and exit
  --dir DIR, -d DIR     Directory containing mesh and trajectory files
  --opath OPATH, -o OPATH
                        Output path. Should be a folder. Files will be written
                        as opath/<framenum>.png
  --frames FRAMES, -f FRAMES
                        frames: either single number n or range n-m
  --tsdf, -t            Whether to use the TSDF mesh or the optimization
                        output.
  --pose POSE, -p POSE  The file storing the camera pose.
```
Most of the arguments are the same as for the C++ executable. However, this
script does not work interactively. We used it to generate figures for the paper
by loading poses saved with the C++ executable.

## Reproducing paper results
For reproducing the results from the paper, you can run the code on sequences 
from the [Co-Fusion dataset](https://github.com/martinruenz/co-fusion).

The [config](config/) directory contains the configurations for all approaches
used in the paper. (Some default parameters are different from EM-Fusion, so
you always need a config file.)

We provide a script [run_cosection.sh](run_cosection.sh) for automatically
running Co-Section on the Co-Fusion datasets (with the datasets downloaded and
extracted as explained in the EM-Fusion readme to `CO-FUSION_DIR`):
```
./run_cosection.sh $CO-FUSION_DIR $OUTPUT_DIR
```

### Reproducing paper figures
We provide the script [generate_cvpr_images.sh](generate_cvpr_images.sh) to
generate all renderings shown in the qualitative evaluation:
```
./generate_cvpr_images.sh $OUTPUT_DIR $IMAGE_DIR
```

### Numerical evaluation
We provide the script [num_eval.sh](num_eval.sh) for evalutating our method
numerically. It uses the [mesh-evalution](https://github.com/davidstutz/mesh-evaluation)
tool by David Stutz to be built and the path in [num_eval.sh](num_eval.sh) set
accordingly. Furthermore, since the evaluation tool only accepts OFF files, we
need to convert the output PLY files first. This is done in the script using
[MeshLab](https://www.meshlab.net/).

We further provide the script [eval_meshes.m](eval_meshes.m) for generating the
visualizations in Figure 6 in the paper. You will need to adapt the result_path
variable to where you stored the Co-Section results.

#### Code change to paper experiments
The code version that created the paper results unintentionally replaced
the normals computed from the pointcloud by those returned from raycasting when
rendering the models. We changed this in this version of the code, resulting in
slightly different numerical results. To achieve numerical results that match
those reported in the paper, configure the project with
`-DUSE_RAYCAST_NORMAL=ON`.

## License
Co-Section has been developed at the [Embodied Vision Group](https://ev.is.mpg.de) at the Max Planck Institute for Intelligent Systems, Germany. The open-source version is licensed under the [GNU General Public License v3 (GPLv3)](LICENSE).

For commercial inquiries, please send email to [ev-license@tue.mpg.de](mailto:ev-license@tue.mpg.de).

