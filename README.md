# Lake Erie Harmful Algal Blooms Analysis

Lake Erie’s Algae Crisis: Analyzing Public Discourse for Sustainable Solutions

Alex Thompson & Bangzhao Shu

This repository contains data and Jupyter notebooks for a comprehensive analysis of harmful algal blooms (HABs) in Lake Erie, focusing on environmental data analysis, correlation studies between environmental factors and social media discussions, and the creation of interactive visualizations.

-------------------------------------------------------------------------------------------------------------------------------------

## Structure of Repository

### Environmental:

  - **Geoheatmap_interactive.html:** An interactive heatmap visualizing microcystin concentrations in different areas of Lake Erie.
  - **Shareable_All Stone Lab Lake Erie data.csv:** Dataset used in analyses.
  - **Stonelab_Lake_Erie_Final.ipynb:** Jupyter notebook for environmental data preprocessing, analysis, and visualization. This is the main file in the Environmental folder.

### SocialMedia:
  
  - **text-classification-llama2:** Python script for classifying environment-related comments using Llama2 via zero-shot learning.
  - **topic-detction-gpt3.5.ipynb:** Jupyter notebook for topic classification using the OpenAI API via zero-shot learning.
  - **stance-detection.ipynb:** Jupyter notebook for stance detection using the OpenAI API via zero-shot learning.

### Correlation:
  
  - **SI699FinalProject_Correlations.ipynb:** Jupyter notebook analyzing correlations between social media and environmental data.
  - **TNTP_monthly_mean.csv:** Monthly averages of TN:TP ratios, derived from environmental analyses.
  - **algal_monthly_mean.csv:** Monthly averages of algal biomass.
  - **microcystin_monthly_mean.csv:** Monthly averages of microcystins.

### Outside the folders:
  
  - **requirements.txt:** Contains a list of Python packages required to run the notebooks.
  - **README.md:** This file.

-------------------------------------------------------------------------------------------------------------------------------------

## Setup

  - **Clone the Repository:** git clone https://github.com/your-username/your-repository.git
  - **Install Required Libraries:** pip install -r requirements.txt
  - **Navigate to the Notebook:** Open the Jupyter notebooks in the respective folders to view the analyses.

-------------------------------------------------------------------------------------------------------------------------------------

## Usage

  - Run 'Stonelab_Lake_Erie_Final.ipynb' to generate the ecological data visualizations and analyses.
  - Enter the path of your Llama2 model, and run 'text-classification-llama2' to detect environment-related comments from the all_RC.json dataset.
  - Enter your OpenAI API key, and run 'topic-detction-gpt3.5.ipynb' or 'stance-detection.ipynb' to do topic classification and stance detection respectively via zero-shot learning.
  - The correlation studies can be explored running 'SI699Final Project_Correlations.ipynb', which utilizes environmental and social media dataset to find patterns and insights.

-------------------------------------------------------------------------------------------------------------------------------------

## Contributions

  - **Environmental Analysis:** Focused on data collection, preprocessing, and detailed environmental analysis.
  - **Social Media Analysis:** Focused on Reddit data retreival, preprocessing, text classification, topic modeling and stance detection.
  - **Correlation Analysis:** Explores how environmental changes correlate with public discourse and sentiment on social media platforms.

-------------------------------------------------------------------------------------------------------------------------------------

## License

  This project is open source and available under the MIT License.
  
