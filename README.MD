
# Buy-The-Look Recommendation System

This project presents a "Buy-The-Look" recommendation system designed to enhance the online shopping experience by suggesting fashion items that complement a user's selected product. By analyzing product images and metadata, the system recommends a cohesive ensemble, enabling users to "buy the look" effortlessly.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

The "Buy-The-Look" recommendation system leverages image processing and machine learning techniques to identify fashion items within a selected product's image and recommend complementary items. This approach aims to replicate the expertise of fashion stylists by providing users with suggestions that create a harmonious outfit.

## Features

- **Product Detection**: Identifies individual fashion items within a product image.
- **Feature Extraction**: Extracts visual features from detected items to understand style and attributes.
- **Recommendation Engine**: Suggests complementary fashion items based on extracted features and existing product metadata.
- **User Interface**: Provides a graphical interface for users to input product images and view recommendations.

## Dataset

The system utilizes a dataset containing images and metadata of fashion products. The dataset includes information such as product categories, styles, and visual attributes, which are essential for training the recommendation model.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/kamisaberi/buy-the-look-recommendation-system.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd buy-the-look-recommendation-system
   ```

3. **Install Required Packages**:

   Ensure that you have Python installed. Then, install the necessary packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**:

    - Ensure that the dataset is available in the project directory. If not, use the `fake-data-generator.py` script to generate sample data:

      ```bash
      python fake-data-generator.py
      ```

2. **Run the User Interface**:

    - Execute the GUI script to launch the application:

      ```bash
      python gui.py
      ```

    - Alternatively, for different versions of the interface, run `gui2.py` or `gui3.py`:

      ```bash
      python gui2.py
      ```

      or

      ```bash
      python gui3.py
      ```

3. **Interact with the Application**:

    - Upload a product image through the interface.
    - View the recommended complementary fashion items.

## Project Structure

```
buy-the-look-recommendation-system/
├── 01.ipynb
├── 02.ipynb
├── 03.ipynb
├── 04.ipynb
├── fake-data-generator.py
├── gui.py
├── gui2.py
├── gui3.py
├── looks.csv
└── products.csv
```

- **01.ipynb - 04.ipynb**: Jupyter notebooks containing exploratory data analysis and model development.
- **fake-data-generator.py**: Script to generate sample data for testing purposes.
- **gui.py, gui2.py, gui3.py**: Different versions of the graphical user interface for the recommendation system.
- **looks.csv**: Dataset containing information about various fashion looks.
- **products.csv**: Dataset containing information about individual fashion products.

## Contributing

Contributions to enhance the "Buy-The-Look" recommendation system are welcome. Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License.
