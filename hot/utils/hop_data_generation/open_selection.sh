#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python hop_data_generator.py --obj_name box --asset_path ../../data/assets/urdf/Box/Box.urdf
python hop_data_generator.py --obj_name sword --asset_path ../../data/assets/urdf/Sword/Sword.urdf
python hop_data_generator.py --obj_name bottle --asset_path ../../data/assets/urdf/Bottle/Bottle.urdf
python hop_data_generator.py --obj_name hammer --asset_path ../../data/assets/urdf/Hammer/Hammer.urdf
python hop_data_generator.py --obj_name shoe --asset_path ../../data/assets/urdf/Shoe/Shoe.urdf
python hop_data_generator.py --obj_name usb --asset_path ../../data/assets/urdf/USB/USB.urdf
python hop_data_generator.py --obj_name book --asset_path ../../data/assets/urdf/Book/Book.urdf
python hop_data_generator.py --obj_name bowl --asset_path ../../data/assets/urdf/Bowl/Bowl.urdf
python hop_data_generator.py --obj_name mug --asset_path ../../data/assets/urdf/Mug/Mug.urdf
python hop_data_generator.py --obj_name pencil --asset_path ../../data/assets/urdf/Pencil/Pencil.urdf
python hop_data_generator.py --obj_name pliers --asset_path ../../data/assets/urdf/Pliers/Pliers.urdf
python hop_data_generator.py --obj_name screwdriver --asset_path ../../data/assets/urdf/Screwdriver/Screwdriver.urdf
python hop_data_generator.py --obj_name stick --asset_path ../../data/assets/urdf/Stick/Stick.urdf
python hop_data_generator.py --obj_name wineglass --asset_path ../../data/assets/urdf/Wineglass/Wineglass.urdf
python hop_data_generator.py --obj_name gun --asset_path ../../data/assets/urdf/Gun/Gun.urdf
python hop_data_generator.py --obj_name pan --asset_path ../../data/assets/urdf/Pan/Pan.urdf
python hop_data_generator.py --obj_name ball --asset_path ../../data/assets/urdf/Ball/Ball.urdf



