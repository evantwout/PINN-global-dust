import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata

params_plot = {
    'axes.labelsize': 25,
    'axes.titlesize': 30,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'lines.linewidth': 20,
    'legend.fontsize': 20,
    'font.family': 'DeJavu Serif',
    'font.serif': 'Times New Roman',
}

plt.rcParams.update(params_plot)


def plot_dust_deposition_map(df_PINN, df_empirical, title, name_to_save, figure_save_path, label_str='log_dep', measure_units='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]', limit_bar=3.2):
    """
    Generate the matplot of the calculation of the PINN.
    Also, the empirical dataset is included in the plot

    Parameters:
    - df_PINN (pandas DataFrame): The dataframe containing the dust deposition predicted by the PINN.
    - df_empirical (pandas DataFrame): The dataframe containing the dust deposition data.
    - title (str): The title to be displayed on the plot.
    - name_to_save (str): The filename to save the generated map plot.
    - label_str (str): The column name in the dataframe to use for labeling the data on the plot. Defaults to 'log_dep'.
    - figure_save_path (str): The directory path where the plot will be saved.
    - measure_units (str): The units of measurement for the dust flux. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'.
    - limit_bar (float,): The limit value for the colorbar on the plot. Defaults to 3.2.
    """

    plt.rcParams.update(params_plot)

    # Change from pandas to geopandas the empirical df for the plot
    df_holocene_empirical_geo = gpd.GeoDataFrame(df_empirical, geometry=gpd.points_from_xy(df_empirical.lon, df_empirical.lat))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot continents
    world.dissolve(by='continent').boundary.plot(ax=ax, color='black', linewidth=0.8)

    # Plot the empirical dataset in geopandas
    df_holocene_empirical_geo.plot(column='log_dep',
                                   marker='o',
                                   ax=ax,
                                   edgecolor='black',
                                   legend=True,
                                   legend_kwds={'label': measure_units, 'orientation': "horizontal"},
                                   cmap='viridis',
                                   vmin=-limit_bar,
                                   vmax=limit_bar,
                                   linewidth=0.5,
                                   markersize=100)

    # Plot the prediction after reshaping it
    U_pred_PINNs_reshape = df_PINN['PINN_log_dep'].values.reshape(60, 120)
    h = ax.imshow(U_pred_PINNs_reshape,
                  origin='lower',
                  extent=[-180, 180, -90, 90],
                  cmap='viridis',
                  vmin=-limit_bar,
                  vmax=limit_bar)

    # Set axis-labels, title, limits and ticks
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_yticks(np.arange(-90, 91, 30))

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()


def plot_scatterplot_simulated(df, name_to_save, x_label, y_label, figure_save_path):
    """
    Generate and save a scatterplot of two specified variables in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        name_to_save (str): Name to use when saving the plot.
        figure_save_path (str): Path to the directory for saving the plot.
    """

    plt.rcParams.update(params_plot)

    # Generate random data
    x = df[['log_dep']].values.reshape(-1)
    y = df[['PINN_log_dep']].values.reshape(-1)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df[['log_dep']].values, df[['PINN_log_dep']].values))

    # Calculate MAE
    mae = mean_absolute_error(df[['log_dep']].values, df[['PINN_log_dep']].values)

    # Calculate kernel density estimation for contour plot
    values = df[["log_dep", "PINN_log_dep"]].values.T
    xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Calculate the levels for contour plot
    min_density = 10**-4
    max_density = 10**0.3

    n_levels = 10  # Number of contour levels
    levels = np.logspace(np.log10(min_density), np.log10(max_density), num=n_levels, endpoint=True)

    # Create a color map and normalization
    cmap = plt.cm.get_cmap('YlOrRd')
    norm = LogNorm(vmin=min_density, vmax=max_density)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the contour plot with logarithmic density scale
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    plt.plot([xmin, xmax], [ymin, ymax], color='black', linewidth=2)

    # Set the color bar with specific tick values and labels
    cbar = plt.colorbar(cf, ax=ax, label="Density", norm=norm)
    cbar_ticks = [val for val in levels]
    cbar.set_ticklabels([f'$10^{{{val:.1f}}}$' for val in np.log10(levels)])

    ax.axis('equal')
    plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    # Add a text box with RMSE and MAE values
    text_str = r"RMSE = {:.2f}".format(rmse) + "\n" + r"MAE = {:.2f}".format(mae)
    ax.text(0.35, 0.95, text_str, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            fontsize=20)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()



def plot_scatterplot_empirical(df, x_variable, y_variable, x_label, y_label, name_to_save, figure_save_path):
    """
    Generate and save a scatterplot of two specified variables in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_variable (str): Name of the column for the x-axis.
        y_variable (str): Name of the column for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        name_to_save (str): Name to use when saving the plot.
        figure_save_path (str): Path to the directory for saving the plot
    """

    plt.rcParams.update(params_plot)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(df[[x_variable]].values, df[[y_variable]].values))

    # Calculate MAE
    mae = mean_absolute_error(df[[x_variable]].values, df[[y_variable]].values)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=df, x=x_variable, y=y_variable)
    X_plot = np.linspace(-4, 4, 100)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color='black', linewidth=2)
    ax.axis('equal')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add a text box with RMSE and MAE values
    text_str = r"RMSE = {:.2f}".format(rmse) + "\n" + r"MAE = {:.2f}".format(mae)
    ax.text(0.35, 0.95, text_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=20)

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()


def plot_hist(df, title, name_to_save, figure_save_path, label_str='PINN_log_dep', label_hist_horizontal='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'):
    """
    Generate and save a histogram plot of a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title of the plot.
        name_to_save (str): Name to use when saving the plot.
        label_str (str): Column label for the histogram. Defaults to 'log_dep'.
        figure_save_path (str): Path to the directory for saving the plot
        label_hist_horizontal (str): Horizontal label for the histogram. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'
    """

    plt.rcParams.update(params_plot)

    # Create a figure and set the x-axis limits
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-3.5, 3.5)

    # Plot the histogram
    df[label_str].plot.hist(grid=True, bins=28, rwidth=0.9, color='grey', density=True, range = [-3.5, 3.5])

    # Fit a normal distribution to the data
    mu, std = norm.fit(np.array(df[label_str], dtype=float))
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Plot the fitted normal distribution
    plt.plot(x, p, 'k', linewidth=2)

    # Set the plot title and labels
    plt.title(title)
    plt.xlabel(label_hist_horizontal)

    # Set the y-axis limits
    ax.set_ylim(0, 0.7)

    # Add gridlines to the y-axis
    plt.grid(axis='y', alpha=0.75)

    # Add a text box with mu and sigma values
    text_str = r"$\mu$ = {:.2f}".format(mu) + "\n" + r"$\sigma$ = {:.2f}".format(std)
    ax.text(0.2, 0.95, text_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=20)

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()

    return

def plot_dust_deposition_simulated(df, title, name_to_save, figure_save_path, label_str='log_dep', measure_units='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]', limit_bar = 3.2):
    """Generate a map plot of dust deposition based on the provided dataframe.

    Parameters:
    - df (pandas DataFrame): The dataframe containing the dust deposition data.
    - title(str): The title to be displayed on the plot.
    - name_to_save (str): The filename to save the generated map plot.
    - label_str (str): The column name in the dataframe to use for labeling the data on the plot. Defaults to 'log_dep'.
    - figure_save_path (str): The directory path where the plot will be saved.
    - measure_units (str): The units of measurement for the dust flux. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'
    - limit_bar (float): The limit value for the colorbar on the plot. Defaults to 3.2."""

    plt.rcParams.update(params_plot)

    df_dust_geopandas = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Set limits and ticks
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_yticks(np.arange(-90, 91, 30))

    # Plot continents
    world.dissolve(by='continent').boundary.plot(ax=ax, color='black', linewidth=0.8)

    # Plot the GeoDataFrame
    df_dust_geopandas.plot(column=label_str,
                           ax=ax,
                           legend=True,
                           legend_kwds={'label': measure_units, 'orientation': "horizontal"},
                           cmap='viridis',
                           vmin=-limit_bar,
                           vmax=limit_bar,
                           linewidth=0.005,
                           markersize=100)

    # Set labels and title
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title)

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()

    return


def plot_dust_deposition_map_zoom(df, title, name_to_save, figure_save_path, label_str, measure_units='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]', limit_bar=3.2):
    """
    Generate a map plot of dust deposition based on the provided dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the dust deposition data.
    - title (str): The title to be displayed on the plot.
    - name_to_save (str): The filename to save the generated map plot.
    - label_str (str): The column name in the dataframe to use for labeling the data on the plot. Defaults to 'log_dep'.
    - figure_save_path (str): The directory path where the plot will be saved.
    - measure_units (str): The units of measurement for the dust flux. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'.
    - limit_bar (float): The limit value for the colorbar on the plot. Defaults to 3.2.
    """

    params_zoom = {
        'axes.labelsize': 35,
        'axes.titlesize': 40,
        'xtick.labelsize': 35,
        'ytick.labelsize': 35
    }

    plt.rcParams.update(params_zoom)

    df_dust_geopandas = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    fig, ax = plt.subplots(figsize=(15, 8))

    # Set limits and ticks
    ax.set_xlim(-90, 90)  # Adjusted longitude limits
    ax.set_ylim(-70, -10)  # Adjusted latitude limits
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_yticks(np.arange(-70, 0, -10))

    # Plot continents
    world.dissolve(by='continent').boundary.plot(ax=ax, color='black', linewidth=0.8)

    # Filter longitude
    df_dust_geopandas = df_dust_geopandas[(df_dust_geopandas["lat"] > -70) & (df_dust_geopandas["lat"] < -10) &
                                          (df_dust_geopandas["lon"] > -90) & (df_dust_geopandas["lon"] <= 90)]

    # Plot the GeoDataFrame using imshow
    U = df_dust_geopandas[label_str].values.reshape(len(df_dust_geopandas['lat'].unique()), len(df_dust_geopandas['lon'].unique()))
    h = ax.imshow(U,
                  origin='lower',
                  extent=[-90, 90, -70, -10],  # Adjusted extent for latitude and longitude
                  cmap='viridis',
                  vmin=-limit_bar,
                  vmax=limit_bar,
                  aspect='auto')  # Maintain the aspect ratio of the plot

    # Add colorbar
    cbar = plt.colorbar(h, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label(measure_units)

    # Set labels and title
    ax.set(xlabel='Longitude', ylabel='Latitude', title=title)
    ax.set_xticks(np.arange(-90, 91, 15))
    ax.set_yticks(np.arange(-70, 0, 10))

    # Save and show the plot
    plt.savefig(f"{figure_save_path}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()
