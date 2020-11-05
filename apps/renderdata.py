#
# This file is part of Co-Section.
#
# Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
# Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
# For more information see <https://cosection.is.tue.mpg.de/>.
# If you use this code, please cite the respective publication as
# listed on the website.
#
# Co-Section is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Co-Section is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Co-Section.  If not, see <https://www.gnu.org/licenses/>.
#
import bpy
import bpy_extras
#import utils
import mathutils

import argparse
import math
import os
import sys
import random


colors = [[255, 255, 255],
          [255, 12, 0],
          [0, 219, 255],
          [175, 0, 255],
          [255, 233, 0],
          [0, 22, 255],
          [0, 255, 64],
          [0, 255, 141],
          [0, 255, 213],
          [255, 227, 0],
          [0, 255, 201],
          [223, 0, 255],
          [255, 167, 0],
          [199, 0, 255],
          [255, 0, 149],
          [0, 255, 177],
          [0, 58, 255],
          [255, 0, 197],
          [0, 255, 129],
          [44, 255, 0],
          [255, 24, 0],
          [0, 94, 255],
          [255, 215, 0],
          [86, 255, 0],
          [0, 255, 22],
          [253, 255, 0],
          [0, 255, 124],
          [0, 28, 255],
          [229, 255, 0],
          [211, 0, 255],
          [255, 191, 0],
          [20, 255, 0],
          [217, 255, 0],
          [0, 16, 255],
          [247, 255, 0],
          [56, 255, 0],
          [127, 255, 0],
          [2, 255, 0],
          [104, 0, 255],
          [38, 255, 0],
          [255, 114, 0],
          [0, 255, 171],
          [0, 231, 255],
          [68, 255, 0],
          [145, 255, 0],
          [157, 255, 0],
          [255, 0, 131],
          [151, 255, 0],
          [0, 100, 255],
          [0, 118, 255],
          [0, 255, 100],
          [62, 255, 0],
          [211, 255, 0],
          [0, 141, 255],
          [0, 255, 94],
          [44, 0, 255],
          [255, 0, 227],
          [0, 255, 70],
          [116, 0, 255],
          [0, 255, 58],
          [255, 0, 203],
          [0, 207, 255],
          [0, 255, 255],
          [0, 171, 255],
          [26, 255, 0],
          [110, 255, 0],
          [163, 0, 255],
          [0, 243, 255],
          [255, 0, 221],
          [0, 237, 255],
          [255, 0, 12],
          [255, 161, 0],
          [255, 48, 0],
          [193, 0, 255],
          [255, 108, 0],
          [0, 255, 82],
          [255, 36, 0],
          [0, 201, 255],
          [0, 255, 10],
          [0, 255, 118],
          [0, 255, 106],
          [255, 0, 36],
          [255, 0, 42],
          [255, 102, 0],
          [0, 153, 255],
          [0, 195, 255],
          [0, 255, 34],
          [20, 0, 255],
          [0, 4, 255],
          [199, 255, 0],
          [0, 255, 231],
          [0, 112, 255],
          [26, 0, 255],
          [32, 0, 255],
          [145, 0, 255],
          [255, 0, 60],
          [229, 0, 255],
          [241, 0, 255],
          [0, 106, 255],
          [98, 0, 255],
          [255, 0, 126],
          [205, 255, 0],
          [0, 159, 255],
          [0, 255, 189],
          [139, 255, 0],
          [8, 255, 0],
          [255, 0, 251],
          [255, 96, 0],
          [253, 0, 255],
          [255, 0, 96],
          [255, 0, 30],
          [14, 255, 0],
          [205, 0, 255],
          [50, 0, 255],
          [255, 0, 179],
          [255, 0, 84],
          [0, 52, 255],
          [0, 255, 40],
          [139, 0, 255],
          [187, 0, 255],
          [0, 255, 219],
          [0, 147, 255],
          [98, 255, 0],
          [2, 0, 255],
          [193, 255, 0],
          [255, 0, 120],
          [0, 40, 255],
          [255, 0, 137],
          [0, 255, 16],
          [187, 255, 0],
          [0, 255, 243],
          [0, 255, 28],
          [0, 46, 255],
          [0, 34, 255],
          [169, 0, 255],
          [255, 185, 0],
          [255, 221, 0],
          [255, 18, 0],
          [0, 70, 255],
          [255, 137, 0],
          [68, 0, 255],
          [255, 149, 0],
          [255, 0, 90],
          [241, 255, 0],
          [122, 255, 0],
          [0, 213, 255],
          [255, 131, 0],
          [80, 255, 0],
          [38, 0, 255],
          [255, 251, 0],
          [86, 0, 255],
          [247, 0, 255],
          [255, 90, 0],
          [255, 203, 0],
          [0, 64, 255],
          [0, 255, 249],
          [255, 0, 102],
          [0, 255, 195],
          [169, 255, 0],
          [255, 0, 114],
          [80, 0, 255],
          [92, 0, 255],
          [0, 255, 225],
          [0, 129, 255],
          [255, 126, 0],
          [56, 0, 255],
          [128, 0, 255],
          [32, 255, 0],
          [0, 255, 88],
          [255, 155, 0],
          [0, 165, 255],
          [0, 82, 255],
          [0, 88, 255],
          [255, 0, 108],
          [255, 0, 239],
          [255, 0, 245],
          [255, 0, 173],
          [8, 0, 255],
          [235, 0, 255],
          [0, 255, 153],
          [255, 78, 0],
          [255, 0, 167],
          [255, 0, 66],
          [255, 0, 54],
          [223, 255, 0],
          [12, 11, 11],
          [255, 0, 209],
          [255, 0, 72],
          [163, 255, 0],
          [0, 76, 255],
          [74, 0, 255],
          [74, 255, 0],
          [0, 10, 255],
          [255, 30, 0],
          [151, 0, 255],
          [255, 197, 0],
          [133, 255, 0],
          [0, 255, 46],
          [0, 255, 147],
          [255, 120, 0],
          [175, 255, 0],
          [255, 173, 0],
          [255, 0, 215],
          [181, 0, 255],
          [235, 255, 0],
          [104, 255, 0],
          [255, 54, 0],
          [0, 177, 255],
          [255, 209, 0],
          [0, 255, 135],
          [255, 0, 155],
          [0, 255, 165],
          [255, 0, 185],
          [255, 245, 0],
          [0, 255, 4],
          [255, 60, 0],
          [0, 255, 183],
          [255, 0, 161],
          [255, 239, 0],
          [110, 0, 255],
          [255, 0, 24],
          [255, 0, 48],
          [181, 255, 0],
          [0, 135, 255],
          [0, 124, 255],
          [255, 66, 0],
          [0, 255, 237],
          [255, 0, 6],
          [0, 189, 255],
          [255, 84, 0],
          [92, 255, 0],
          [0, 249, 255],
          [255, 6, 0],
          [14, 0, 255],
          [0, 255, 112],
          [50, 255, 0],
          [255, 143, 0],
          [255, 0, 78],
          [0, 255, 159],
          [133, 0, 255],
          [255, 0, 18],
          [0, 225, 255],
          [255, 0, 143],
          [0, 255, 76],
          [217, 0, 255],
          [157, 0, 255],
          [255, 0, 191],
          [122, 0, 255],
          [255, 179, 0],
          [62, 0, 255],
          [0, 255, 207],
          [0, 255, 52],
          [255, 0, 233],
          [255, 42, 0],
          [255, 72, 0],
          [0, 183, 255]]


def main_loop(dirpath, opath=None, startframe=0, endframe=None, tsdf=False, campose=None):
    bpy.data.objects.remove(bpy.data.objects['Cube'])
    bpy.data.objects.remove(bpy.data.objects['Light'])

    cam = bpy.data.objects['Camera'].data
    w, h = (640, 480)
    fx = fy = 525.
    cx = w/2 - .5
    cy = h/2 - .5

    calfile = os.path.join(dirpath, 'calibration.txt')
    if os.path.exists(calfile):
        with open(calfile, 'r') as f:
            line = f.readline()
            params = line.split(' ')
            fx, fy, cx, cy = map(float, params[:4])
            w, h = map(int, params[4:])

    cam.shift_x = -(cx / w - .5)
    cam.shift_y = (cy - .5 * h) / w

    cam.lens = fx / w * cam.sensor_width

    cam = bpy.data.objects['Camera']
    cam.location = mathutils.Vector((0,0,0))
    cam.rotation_euler = mathutils.Vector((math.pi,0,0))
    if campose is not None:
        loc, rot, _ = campose.decompose()
        cam.location = loc
        cam.rotation_euler.rotate(rot)

    light_data = bpy.data.lights.new(name='Light', type='AREA')
    light_data.energy = 500
    light_data.size = 10.24
    light = bpy.data.objects.new(name='Light', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0, -2.56, 2.56)
    light.rotation_euler = (1/2*math.pi, 0, 0)

    colormap = [mathutils.Color([c / 255 for c in rgb]) for rgb in colors]

    pixel_aspect = fy / fx
    scene = bpy.context.scene
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = pixel_aspect

    camposes = {}
    cam_posefile = os.path.join(dirpath, 'poses-cam.txt')
    lastframe = -1
    with open(cam_posefile, 'r') as f:
        for line in f:
            parts = line.split(' ')
            t = int(parts[0])
            if t < startframe:
                continue

            loc = mathutils.Vector(map(float, parts[1:4]))
            rot = mathutils.Quaternion(map(float, [parts[i] for i in [7,4,5,6]]))
            camposes[t] = {'loc': loc, 'rot': rot}
            lastframe = t
            if t == endframe:
                break

    poses = {}
    obj = 1
    while True:
        posefile = os.path.join(dirpath, 'poses-' + str(obj) + '.txt')
        if not os.path.exists(posefile):
            break

        poses[obj] = {}
        with open(posefile) as f:
            for line in f:
                parts = line.split(' ')
                t = int(parts[0])
                if t < startframe:
                    continue

                loc = mathutils.Vector(map(float, parts[1:4]))
                rot = mathutils.Quaternion(map(float, [parts[i] for i in [7,4,5,6]]))
                poses[obj][t] = {'loc': loc, 'rot': rot}
        obj += 1

    frame = startframe
    last_keyframe = None
    meshes = {}
    while True:
        if campose is None:
            cam.location = camposes[frame]['loc']
            cam.rotation_euler = mathutils.Vector((math.pi,0,0))
            cam.rotation_euler.rotate(camposes[frame]['rot'])
        newFrame = frame if tsdf else (frame // 10) * 10
        if newFrame != last_keyframe:
            last_keyframe = newFrame
            for obj in bpy.data.objects:
                if obj.name != 'Light' and obj.name != 'Camera':
                    bpy.data.objects.remove(obj)
                meshes.clear()
            if tsdf:
                bg_meshfile = os.path.join(dirpath, 'frame_meshes', 'bg', '%04d.ply' % last_keyframe)
            else:
                bg_meshfile = os.path.join(dirpath, 'optim_meshes', 'bg', '%04d.ply' % last_keyframe)
            bpy.ops.import_mesh.ply(filepath=bg_meshfile)
            bg_mesh = bpy.context.selected_objects[0]
            bg_mesh.name = "BG"
            bg_mesh.location = mathutils.Vector((0, 0, 2.56))
            mat = bpy.data.materials.new(name="Mat bg")
            mat.use_nodes = True
            bg_mesh.data.materials.append(mat)
            #mat.diffuse_color = (.3, .3, .3, 1.0)
            #for f in bg_mesh.data.polygons:
                #f.use_smooth = True

            for obj in range(1, len(poses) + 1):
                if tsdf:
                    meshfile = os.path.join(dirpath, 'frame_meshes', str(obj), '%04d.ply' % last_keyframe)
                else:
                    meshfile = os.path.join(dirpath, 'optim_meshes', str(obj), '%04d.ply' % last_keyframe)

                if os.path.exists(meshfile):
                    bpy.ops.import_mesh.ply(filepath=meshfile)
                    meshes[obj] = bpy.context.selected_objects[-1]
                    meshes[obj].name = str(obj)
                    mat = bpy.data.materials.new(name="Mat " + str(obj))
                    mat.diffuse_color = (colormap[obj].b, colormap[obj].g, colormap[obj].r, 1)
                    mat.use_nodes = True
                    meshes[obj].data.materials.append(mat)
                    meshes[obj].rotation_mode = 'QUATERNION'
                    #for f in meshes[obj].data.polygons:
                        #f.use_smooth = True

        for obj, mesh in meshes.items():
            mesh.location = poses[obj][frame]['loc']
            mesh.rotation_quaternion = poses[obj][frame]['rot']

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        bpy.context.scene.render.tile_x = 256
        bpy.context.scene.render.tile_y = 256
        bpy.context.scene.view_layers['View Layer'].cycles.use_denoising = True
        bpy.context.scene.render.filepath = os.path.join(opath, '%04d.png' % frame)
        bpy.ops.render.render(write_still=True)
        frame += 1
        if frame > lastframe:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D models created by "
                                                 "EM-Fusion or Co-Section")
    parser.add_argument(
        "--dir",
        "-d",
        dest="dir",
        required=True,
        help="Directory containing mesh and trajectory files"
    )
    parser.add_argument(
        "--opath",
        "-o",
        dest="opath",
        required=True,
        help="Output path. Should be a folder. Files will be written as opath/<framenum>.png"
    )
    parser.add_argument(
        "--frames",
        "-f",
        dest="frames",
        help="frames: either single number n or range n-m"
    )
    parser.add_argument(
        "--tsdf",
        "-t",
        dest="tsdf",
        action="store_true",
        default=False,
        help="Whether to use the TSDF mesh or the optimization output."
    )
    parser.add_argument(
        "--pose",
        "-p",
        dest="pose",
        help="The file storing the camera pose."
    )
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    #argv = utils.extract_args()
    args = parser.parse_args(argv)

    startframe = 0
    endframe = None

    if args.frames is not None:
        parts = args.frames.split("-")
        startframe = int(parts[0])
        if len(parts) > 1:
            endframe = int(parts[1])
        else:
            endframe = startframe

    campose = None
    if args.pose is not None:
        with open(args.pose, 'r') as f:
            m = [[float(n) for n in line.split()] for line in f]
            campose = mathutils.Matrix(m)

    main_loop(args.dir, args.opath, startframe, endframe, args.tsdf, campose)
