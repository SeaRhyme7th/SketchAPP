# SketchAPP

A simple Sketch-Based Image Retrieval web application. The system uses Flask for web application framework, and PyTorch for retrieving.

## Dependency

- pytorch 0.4.0 with torchvision 0.2.1 or higher (test with pytorch 1.5.0+cpu and torchvision 0.6.0+cpu)
- python 3.6.4
- flask 0.12.2
- anaconda 4.4.10 recommend

## Retrieving

The system applies Siamese network based on T.Bui et al's paper "[Compact Descriptors for Sketch-based Image Retrieval using a Triplet loss Convolutional Neural Network](https://doi.org/10.1016/j.cviu.2017.06.007)"[[Repo](https://github.com/TuBui/Triplet_Loss_SBIR)|[Page](http://www.cvssp.org/data/Flickr25K/CVIU16.html)]. All retrieving code comes from [jjkislele/SketchTriplet](https://github.com/jjkislele/SketchTriplet).

## Web application demo

The web application has 2 pages, including 'canvas' and result 'panel'. 

In 'canvas' page, you can draw freehand sketch on canvas. 6 icon buttons lie at the bottom of the page. Each button implies its function by its shape: the first pencil-like button is used for drawing. The second eraser-like button is used to clean lines. The third brush-like button is used to clear the screen. The back button is used to withdraw by step. The diskette-like button is used to store the sketch when you complete your drawing. The last upload button is used to load your local sketch file if you don't want to draw by yourself. The right-side slider is used for changing your stroke size.

In 'panel' page, you can get 18 per page, 90 in total retrival results. Every retrieval result has its name with classfication.

![example](https://github.com/SeaRhyme7th/SketchAPP/blob/master/example.gif)

## How to deploy

Run `download_prerequisites.sh` to download the dataset, the offline feature of the dataset, and the retrieval model from Google drive. 

- offline feature [Google drive](https://drive.google.com/open?id=1Z1fbJrNnjD7VrubYBCtfweYNoMjH1uEW)
- dataset [Google drive](https://drive.google.com/open?id=1PqzIO-OWTeEAl3Hs5tRavRs6-qZ8OmXb)
- retrieval model [Google drive](https://drive.google.com/open?id=1oUDCTENBzdBok7rjB_B8zHE3mwdw_0ve)

Put them in the right place. And run python script

```bash
python controller.py
```

Open your Internet browser. Visit `http://localhost:5000/canvas`, and enjoy it!

## Special thanks

1. Canvas designer [@zhoushuozh](https://github.com/zhoushuozh/drawingborad)
2. Retieval model [@TuBui](https://github.com/TuBui/Triplet_Loss_SBIR)
