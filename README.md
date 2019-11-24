# Diver identification for Human-robot collaboration

### Usage
- train your diver detection model by Faster R-CNN according to this [tutorial](https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce)
- under the tensorflow/models/research/object_detecion/detection_obj.py, set the path to your labels, test images and frozen training graphs
- run python detection_obj.py
- under the path to your test images folder, the final training results will be shown (i.e., override the original images).
- If you need pretrained model, please send the author Youya Xia (youyaxia@cs.cornell.edu) an email request.

### Citation
- Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8794290
- Bibliography entry for citation:
```
@inproceedings{xia2019visual,
  title={Visual Diver Recognition for Underwater Human-Robot Collaboration},
  author={Xia, Youya and Sattar, Junaed},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={6839--6845},
  year={2019},
  organization={IEEE}
}
```
### Acknowledgements
- https://github.com/tensorflow/models
