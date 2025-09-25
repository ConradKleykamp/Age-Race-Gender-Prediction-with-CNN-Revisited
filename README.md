# Age-Race-Gender-Prediction-with-CNN

This repository includes the Python notebook (originally completed on Kaggle) that was completed as the final project for CU Boulder's Introduction to Deep Learning course. 

![image](https://github.com/user-attachments/assets/34706b04-fab7-4b99-8f89-baf540a933ce)

---

### Objective 
This is an updated version of my original attempt at this project. Over the past few years, my grasp of both data science and machine learning techniques has significantly improved. Because of this, I thought it would be a great learning experience to come back to this project and improve upon it. This version will contain:

- More streamlined, explainable code
- A better executed EDA
- A more accurate CNN model (including age, gender, AND race!)

The objective of this project is to develop a convolutional neural network (CNN) that will work to predict the age, gender, and race of the faces found in UTKFace dataset. The accuracy and loss of the CNN model will be analyzed. Lastly, the trained CNN model will work to predict the age and gender on several example images.

UTKFace, is available both on Kaggle and on Github. This dataset is a large-scale face dataset with faces ranging from 0 to 116 years old. This dataset contains over 20,000 face images. Each image includes only a single face, and includes annotations of age, gender, and ethnicity. Age is an integer from 0 to 116, gender is either 0 (male) or 1 (female), and race is an integer from 0 to 4, i.e., White (0), Black (1), Asian (2), Indian (3), Others (4). These images also contain variation in pose, facial expression, illumination, occlusion, and resolution. This dataset is available for non-commerical research purposes only.

---

### Methods
Libraries Used
- numpy
- pandas
- random
- matplotlib
- seaborn
- pathlib
- os
- scipy
- PIL (Image)
- tensorflow (keras)
- sklearn (train_test_split)

Exploratory Data Analysis (EDA)
- Plotting the distributions of gender, age, and race
  - Age (histogram with kde line)
  - Gender (Countplot)
  - Race (Countplot)
- Skewness

Data Preprocessing
- Binning age into 4 categories
- Train/test split, stratifying data by gender+race
- Random augmentations when training (horizontal flip, brightness, contrast)

CNN Layers
- Total parameters: 5,713,930
- Trainable parameters: 5,713,802
- Three convolutional layers
- One batch normalization layer
- Three MaxPooling2D layers
- One flatten layer
- Two dense layers
- One dropout layers
- Adam optimizer
  - Gender --> loss: binary_crossentropy, output: accuracy
  - Age --> loss: sparse_categorical_crossentropy, output: accuracy
  - Race --> loss: sparse_categorical_crossentropy, output: accuracy

Model Fitting and Training
- 15 epochs

---

### Conclusion

**Gender Performance**

Gender accuracy for both train and validation sets increased across epochs (albeit with decreases at epochs 2, 3, and 6). The final gender accuracy scores were:

- 0.8915 (train)
- 0.8834 (validation)

Loss remained low and decreased marginally across epochs (with the exceptions of 2 and 6) for the training data. Validation loss was erratic, with significant increases from 0 to 6, but significant decreases 6 to 15. The final gender losses were:

- 0.2511 (train)
- 1.6084 (validation)

**Age Performance**

Similar to gender, age accuracy increased across epochs but had dips at epochs 3 and 6. The final age accuracy scores were:

- 0.7516 (train)
- 0.7490 (validation)

The losses for age were very similar to that of gender, with a stable, slight decrease in training loss but erratic behavior for the validation loss. The validation loss was consistently much higher than the training validation. The final age losses were:

- 0.5615 (train)
- 4.5490 (validation)

**Race Performance**

Race accuracies followed the same pattern, with both training and validation accuracies increasing across epochs but dipping slightly at epochs 3 and 6. The final race accuracies were:

- 0.7792 (train)
- 0.7619 (validation)

The losses followed the same patterns as above. The final race losses were:

- 0.6371 (train)
- 5.2616 (validation)

**Summary of Findings**

Overall, this iteration of this project has shown much improvement over my original work. Specifically, the model is now far better at predicting age (~75% accuracy vs ~4%!) and can now also predict race. All predictions (gender, age, race) faired much better than random guess, which shows that this model is progressing in the right direction. 

Despite the improvements, the drastic fluctuations in the validation losses suggest that the model is likely overfitting the data. 

Future iterations of this project could include the following adjustments:

- Adding more layers to the CNN model
- Adding more regularization
- Using prebuilt models, such as EfficientNetB0
- Different binning techniques for age
- Label smoothing for age
