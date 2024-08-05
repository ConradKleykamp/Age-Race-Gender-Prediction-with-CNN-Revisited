# Age-and-Gender-Prediction-with-CNN

This repository includes the Python notebook (originally completed on Kaggle) that was completed as the final project for CU Boulder's Introduction to Deep Learning course. 

![image](https://github.com/user-attachments/assets/34706b04-fab7-4b99-8f89-baf540a933ce)

---

### Objective 
The objective of this project is to leverage the teachings of CU Boulder's Introduction to Deep Learning course and develop a convolutional neural network (CNN) that will predict the age and gender of various faces. The accuracy and loss of the CNN model will be analyzed. The dataset, called UTKFace, is availale on both Kaggle and Github. The dataset is a large-scale face dataset with faces ranging from 0 to 116 years old. The dataset contains over 20,000 face images. Each image contains a single face and includes annotations of age, gender, and ethnicity. Age is an integer from 0 to 116, gender is either 0 (male) or 1 (female), and race is an integer from 0 to 4, i.e. White (0), Black (1), Asian (2), Indian (3), Others (4). These images also contain variation in pose, facial expression, illumination, occlusion, and resolution. The dataset is available for non-commercial research purposes only. 

---

### Methods
Data Preprocessing
- Mapping 'gender' = 0 to "Male" and 'gender' = 1 to "Female"
- Assigning 'age' as dtype 'float32'
- Assigning 'gender' as dtype 'int32'
- Training set (80%), testing set (20%)
- Normalize the data to reduce the computational footprint of the project

Exploratory Data Analysis (EDA)
- Plotting the distributions of 'age' and 'gender' with bar charts
- Seaborn

CNN Layers
- Total parameters: 11,020,546
- Trainable parameters: 11,020,418
- Three convolutional layers
- One batch normalization layer
- Three MaxPooling2D layers
- One flatten layer
- Three dense layers
- Two dropout layers
- Adam optimizer w/ binary crossentropy and MAE loss

Model Fitting and Training
- 20 epochs
- Batch size = 10
- Validation split = 0.1

---

### General Results
![Screen Shot 2024-08-05 at 9 39 09 AM](https://github.com/user-attachments/assets/82675386-f39b-41bb-ab95-5dc875516abd)
- Best Gender Prediction Accuracy (Validation): 0.8935
- Best Gender Prediction Loss (Validation): 0.4530
- Best Age Prediction Accuracy (Validation): 0.0474
- Best Age Prediction Accuracy (Validation): 7.8650

Overall, the final CNN model was extremely effective at predicting gender but was somehow unable to correctly predict age. The drastic fluctuations in the validation loss for both gender and age suggest that the model is likely overfitting the data. As I am using this project as a learning experience, I am still quite happy with the result.

---

### Future Changes
- Reducing the number of convolutional and dense layers in the CNN
- Adding more regularization to the model
- Training on a smaller number of images
