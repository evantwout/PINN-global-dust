params = {'axes.labelsize': 25,
          'axes.titlesize': 30,
          'xtick.labelsize': 25,
          'ytick.labelsize' : 25,
          'lines.linewidth' : 20,
          'legend.fontsize': 20,
          'font.family':'DeJavu Serif',
          'font.serif' :'Times New Roman'}

plt.rcParams.update(params)

def plot_dust_deposition_map(df, title, name_to_save, label_str='log_dep', url_save=FIGURE_PATH, measure_units='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]', limit_bar = 3.2):
    """Generate a map plot of dust deposition based on the provided dataframe.

    Parameters:
    - df (pandas DataFrame): The dataframe containing the dust deposition data.
    - title(str): The title to be displayed on the plot.
    - name_to_save (str): The filename to save the generated map plot.
    - label_str (str, optional): The column name in the dataframe to use for labeling the data on the plot. Defaults to 'log_dep'.
    - url_save (str, optional): The directory path where the plot will be saved. Defaults to FIGURE_PATH.
    - measure_units (str, optional): The units of measurement for the dust flux. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'.
    - limit_bar (float, optional): The limit value for the colorbar on the plot. Defaults to 3.2."""

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
    plt.savefig(f"{url_save}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)
    plt.show()

    return

def plot_hist(df, title, name_to_save, label_str='log_dep', url_save=FIGURE_PATH, label_hist_horizontal='Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'):
    """
    Generate and save a histogram plot of a specified column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        title (str): Title of the plot.
        name_to_save (str): Name to use when saving the plot.
        label_str (str, optional): Column label for the histogram. Defaults to 'log_dep'.
        url_save (str, optional): Path to the directory for saving the plot. Defaults to FIGURE_PATH.
        label_hist_horizontal (str, optional): Horizontal label for the histogram. Defaults to 'Dust flux log$_{10}$[g m$^{-2}$ a$^{-1}$]'.
    """
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
    plt.savefig(f"{url_save}/{name_to_save}.pdf", bbox_inches='tight', dpi=600)

    return
