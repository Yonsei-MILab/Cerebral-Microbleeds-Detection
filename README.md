# Cerebral-Microbleeds-Detection

This project is developed to detect cerebral microbleeds (CMBs) from brain MR images. 

To do so, we have proposed a fully automated two-stage integrated deep learning approach for efficient CMBs detection. 

The detection stage via the regional-based YOLO endeavors to deny the background regions and simultaneously retrieve potential microbleeds candidates. 

The 3D-CNN stage is developed to reduce the false positives and single out the accurate microbleeds.


# CMBs-Candidate-Detection-Via-YOLO:

The source code of this stage is available at: https://pjreddie.com/darknet/yolov2/.

Moreover, we have uploaded the yolov2.cfg file in this repository.

# False-Positive-Reduction-via-3D-CNN:

Here, we make a complete python code for the second stage available for researchers.

The file named "CMB_3DCNN.py" contains all processing steps: data reading, 3D network generation, training and testing the model, and saving the network weights and predicted labels.
You can easily change the hyper-parameters such as learning rate, number of epochs, batch size, and selecting appropriate loss function and optimizer.

# Dataset:
The dataset used in this project has been collected with a collaboration between the Medical Imaging LABoratory (MILAB) at Yonsei University and Gachon University Gil Medical Center.

You can get access to the dataset with labels through the following downloading link:
http://kimchi.yonsei.ac.kr/default/06/01.php

Our dataset contains two in-plane resolutions as follows:
1. High in-plane resolution (HR): 0.50x0.50 mm^2, and
2. Low in-plane resolution (LR): 0.80x0.80 mm^2.

HR data composites of 72 subjects, while LR data contains 107 subjects.
All Data contain SWI, Phase and Magnitude images.

The Label folder involves excel files, where each excel file is with the same name as data in the Data folder.
The information of the location of cerebral microbleeds (CMBs) in brain images exist in those excel files as follow:
- 1st column represents the slice number of subject,
- 2nd and 3rd columns indicate the x(column-wise) and y(row-wise) pixel location of CMB in that slice, respectively.

# Published-Paper:

[1] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim “Automated Detection of Cerebral Microbleeds in MR Images: A Two-Stage Deep Learning Approach,” NeuroImage: Clinical, vol. 28, pp. 102464, 2020. [ELSEVIER Publisher]

[2] Mohammed A. Al-masni, Woo-Ram Kim, Eung Yeop Kim, Young Noh, and Dong-Hyun Kim, “A Two Cascaded Network Integrating Regional–based YOLO and 3D-CNN for Cerebral Microbleeds Detection,” in the 42nd Annual International Conferences of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 1055-1058. IEEE, Montreal, Quebec, Canada, 2020.

When using our code or dataset for research publications, please cite our papers.


