**Facial Expression detection using Compact Convolutional Transformer(CCT)**

**Project Overview**

This project involves building a facial expression detection system using machine learning. The model leverages the Compact Convolutional Transformer (CCT) architecture, which combines convolutional layers and transformers for effective facial expression classification. The model is trained on the FER-2013 dataset, which contains facial images labeled with various emotions such as happy, sad, angry, etc.

**Table of contents**

## **Table of Contents**

- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Process](#process)
- [Model Training](#model-training)
- [Results](#results)

**Project Description**

Facial expression detection is a key application in human-computer interaction. This project uses the Compact Convolutional Transformer (CCT) model, which is known for its ability to capture both local and global features of images effectively. By combining CNNs and transformers, the model achieves high accuracy in classifying facial emotions from images.

The goal of this project is to develop a system that can predict facial expressions in real-time or from static images, with applications in fields like customer service, healthcare, and entertainment.

**Technologies Used**

1. **Programming Language**: Python
2. **Libraries**: 
   1. TensorFlow/Keras for model development and training
   2. OpenCV for image preprocessing and video capture
   3. NumPy for numerical operations and data manipulation
   4. Matplotlib for visualizing results and training performance

3. **Model**: Compact Convolutional Transformer (CCT)
4. **Dataset**: We have collected the database from kaggle. This project uses the FER-2013 dataset, which contains over 35,000 labeled images representing 7 different emotions (angry, disgusted, fearful, happy, sad, surprised, and neutral).

**Project Process**

**1. Defining the Problem**

The main objective of this project was to create a system that can automatically detect human emotions from facial expressions. The emotions to be detected include happy, sad, angry, surprised, neutral, and others. This task is crucial in areas such as human-computer interaction, customer feedback systems, and assistive technologies for people with special needs.

**2. Selecting the Model**

we decided to use the Compact Convolutional Transformer (CCT) model. This architecture is a hybrid of convolutional neural networks (CNNs) and transformers, which allows the model to extract both local features (such as facial landmarks) and global context (such as relationships between facial regions) effectively. We chose this architecture because it strikes a balance between performance and computational efficiency.

**3. Data Collection and Preprocessing**

The FER-2013 dataset, available on Kaggle, was used for this project. This dataset contains grayscale facial images labeled with different emotions. The preprocessing steps included:

Resizing the images to a standard size of 48x48 pixels to ensure uniformity.
Normalizing pixel values to a range between 0 and 1 by dividing the pixel values by 255.
Splitting the dataset into training, validation, and test sets (80% for training, 10% for validation, and 10% for testing).
Augmenting the data with techniques such as rotation, flipping, and zooming to increase the diversity of the training set and reduce overfitting.

**4. Model Architecture**

The CCT model was implemented with the following layers:

Convolutional layers: These extract local features from the facial images.
Transformer layers: These capture the global dependencies between facial features, allowing the model to recognize more complex patterns and relationships.
Fully connected layers: These combine the extracted features and output a vector representation.
Softmax output layer: This layer predicts the probabilities of each emotion class.
The model architecture was designed to be compact and efficient while still achieving high accuracy.

**5. Training the Model**

Once the model architecture was defined, the next step was to train the model. The training process included:

Compiling the model with the Adam optimizer and categorical cross-entropy loss function.
Training the model for 50 epochs, using a batch size of 32. The model was trained on the training set, with validation data used to monitor performance and prevent overfitting.
Evaluating the model on the test set to check its performance in classifying unseen data.
During training, we monitored the model’s accuracy and loss to ensure it was learning effectively.

**6. Model Evaluation and Fine-tuning**

After training the model, we evaluated it on the test dataset. The final evaluation metrics were:

Test Accuracy: The percentage of correct predictions on the test set.
Loss: The measure of error between the predicted and true labels.
Confusion Matrix: This provided detailed insights into how well the model predicted each emotion class.
Based on the results, we fine-tuned hyperparameters such as the learning rate and the number of epochs to improve the model’s performance.

**Results and Conclusion**

The model was able to achieve a test accuracy of 70.21% on the test set, with the most common prediction errors occurring in the (emotion class). However, the model demonstrated good performance across all emotion categories, particularly for (emotion), which showed a higher prediction accuracy. The project demonstrates the potential of hybrid models like CCT in facial expression recognition and can be extended to real-world applications such as customer sentiment analysis, interactive gaming, and assistive technologies.
