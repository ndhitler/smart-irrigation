# Smart Irrigation System Using Machine Learning

## 1. Project Overview

The Smart Irrigation System is a project that uses machine learning to help make irrigation decisions easier. It considers factors such as soil moisture, temperature, and air humidity to predict whether plants may need watering.

The system is developed using Python and Streamlit, with a Random Forest classification model. The main aim is to show how technology can help farmers manage irrigation and avoid unnecessary water use.

## 2. Project Objectives

The main objectives of this project are to:

- Apply machine learning to a real-life agricultural problem.
- Use soil moisture, temperature, and air humidity to support irrigation decisions.
- Develop a simple web application that displays irrigation predictions.
- Demonstrate how smart irrigation can help save water and improve farm management.

## 3. Technologies Used

The following tools and technologies were used to develop the project:

- **Python** – for writing the application and processing data.
- **Streamlit** – for creating the web interface.
- **Pandas** – for handling and organizing data.
- **Scikit-learn** – for building the machine-learning model.
- **Random Forest** – for predicting whether irrigation is needed.

## 4. How the System Works

The system works by taking environmental values such as soil moisture, temperature, and air humidity as inputs. The machine-learning model processes these values and predicts whether irrigation may be required. The result is then displayed on the Streamlit application.

The current version is a prototype that uses simulated data. In the future, real sensors and an automatic water-pumping system could be added to make it more practical.

## 5. Live Application

You can access and test the application using the link below:

[Open the Smart Irrigation Application](https://smart-irrigation-mrpa2ygypvnmq92xe8znew.streamlit.app/)

## 6. Source Code

The project's source code is available on GitHub:

[View the GitHub Repository](https://github.com/ndhitler/smart-irrigation)

## 7. Secondary Dataset Source

For this project, a relevant external dataset is available on Mendeley Data:

Kaur, A., Bhatt, D. P., & Raja, L. (2023). *Soil Moisture, Air temperature, humidity, and Motor on/off Monitoring data* (Version 1). Mendeley Data.

[View the Dataset](https://doi.org/10.17632/fpdwmm7nrb.1)

**Note:** The current application uses simulated data to train its model. The external dataset is listed as a relevant data source, but it has not been confirmed as part of the model's training or testing data.

## 8. How to Run the Project

To run the application on your computer:

1. Install Python.
2. Download or clone the project from GitHub.
3. Open a terminal in the project folder.
4. Install the required libraries listed in `requirements.txt`.
5. Start the application by running:

`streamlit run app.py`

The application should then open in your web browser.

## 9. Limitations and Future Improvements

Although the prototype demonstrates how machine learning can support irrigation decisions, it still has some limitations. It currently uses simulated data rather than readings from real farm sensors.

Possible improvements include:

- Connecting real soil-moisture, temperature, and humidity sensors.
- Training and testing the model with verified real-world data.
- Adding automatic water-pump control.
- Monitoring soil conditions continuously and improving irrigation decisions.

## 10. Project Members

- NDAYIMENYE Samuel
- HABYARIMANA Stanislas

**Institution:** RP Kigali  
**Programme:** BTech in Electronics and Telecommunication Technology
