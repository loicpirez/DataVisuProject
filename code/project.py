import plotly.express as px
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import prediction as pr

filename = "dataset/data.csv"

def main():
    visualization = select_visualization()
    while (visualization not in ["1", "2", "3"]):
        print("Undefined visualization, please select among the possible visualizations.")
        visualization = select_visualization()
    if (visualization == "1"):
        visualization_parallel_coordinates()
    elif (visualization == "2"):
        visualization_categorical_scatterplot()
    elif (visualization == "3"):
        select_region()
    exit(0)

def select_region():
    df = pandas.read_csv(filename)
    saved_column = df.region
    regions = []
    choice = -1
    for region in saved_column.drop_duplicates():
        regions.append(region)
    regions = sorted(regions, key=str.lower)
    for region in regions:
        print(regions.index(region), " - ", region)
    while choice not in range(len(regions)):
        print("----------------------------------------------------")
        print("Select a region:")
        choice = input()
        if (int(choice) in range(len(regions))):
            pr.main(regions[int(choice)])
            exit(0)
    
def visualization_parallel_coordinates():
    labels = {
        "year": "Year",
        "npg": "Natural population growth",
        "birth_rate": "Birth rate",
        "death_rate": "Death rate",
        "gdw": "Demographic weight",
        "urbanization": "Urbanization"
    }

    df = pandas.read_csv(filename, usecols=[
        "year", "birth_rate", "death_rate", "gdw", "urbanization", "npg"])
    fig = px.parallel_coordinates(df,
                                  color="year",
                                  labels=labels,
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2,
                                  range_color=[1990, 2017])
    fig.show()

def visualization_categorical_scatterplot():
    df = pandas.read_csv(filename, usecols=[
        "year", "birth_rate", "death_rate", "gdw", "urbanization", "npg", "region"])
    for i in df.year.unique():
        sns.catplot(x="region", y="npg", data=df[df['year'] == i])
        plt.ylim(-15, 20)
        plt.xticks(fontsize=5)
        plt.xticks(rotation=90)
        plt.savefig(f"render/scatter_plot_{i}.png")
        plt.close()
    sns.catplot(x="region", y="npg", data=df)
    plt.ylim(-15, 20)
    plt.xticks(fontsize=5)
    plt.xticks(rotation=90)
    plt.show()
    plt.close()


def select_visualization():
    print("[Russian Demography Data]")
    print("----------------------------------------------------")
    print("Enter one of the following number :")
    print("* Data visualization")
    print("1 - Display the parallel coordinates")
    print("2 - Generate and display categorical scatterplot")
    print("* Prediction algorithm")
    print("3 - Predict the birth rate for a region")
    print("----------------------------------------------------")
    return input()

if __name__ == "__main__":
    main()
