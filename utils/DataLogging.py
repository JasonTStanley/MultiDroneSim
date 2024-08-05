import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataLogger:

    def __init__(self, file_name='data.csv', start_keys=[]):
        self.data = {} #dictionary of lists
        self.key_abbr = {}
        for key in start_keys:
            self.data[key] = []
        self.file_name = file_name
        self.deriv_dict = {}
        self.current_figure = None
        self.xyz = ['x', 'y', 'z']

    def add_entry(self, key, value, names=['x','y','z']):
        try:
            #handled when value is a vector, adds subscript with index and allows for printing of the parent or children
            if len(list(value)) > 1:
                for i in range(len(value)):
                    full = f"{key}_{names[i]}"
                    if key not in self.key_abbr:
                        self.key_abbr[key] = []
                    if full not in self.key_abbr[key]:
                        self.key_abbr[key].append(full)
                    self.add_entry(full, value[i])
                return
        except:
            pass

        if key not in self.data:
            self.data[key] = []


        self.data[key].append(value)
        if key in self.deriv_dict:
            dt, deriv_name = self.deriv_dict[key]
            if len(self.data[key]) > 1:
                self.data[deriv_name].append((self.data[key][-1] - self.data[key][-2] )  / dt)



    def add_entry_deriv(self, key, dt):
        '''Adds a key to the data dictionary that is the derivative of the key with respect to time'''

        if key not in self.data.keys():
            print(f"Key {key} not found in data")
        else:
            deriv_name = f"d{key}/dt"
            if deriv_name in self.data.keys():
                print(f"Key {deriv_name} already exists in data")
            else:
                self.data[deriv_name] = list(np.diff(self.data[key]) / dt)
            self.deriv_dict[key] = (dt, deriv_name)

    def save_csv(self):
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.data.items()]))
        if self.file_name.split('.')[-1] != 'csv':
            self.file_name += '.csv'
        df.to_csv(self.file_name, index=True, header=True, sep=',')
        print("SAVED CSV FILE TO: ", self.file_name)

    def create_figure(self, figsize=(10, 6)):
        fig = plt.figure()
        self.current_figure = fig
        self.current_figure.set_size_inches(figsize)
        return fig

    def plot_all(self, show=False):
        for key in self.data:
            self.plot_key(key, show=False)
        if show:
            plt.legend()
            plt.show()

    def plot_key(self, key, show=False):
        if key in self.key_abbr:
            for full in self.key_abbr[key]:
                plt.plot(self.data[full], label=full)
        else:
            plt.plot(self.data[key], label=key)

        if show:
            plt.legend()
            plt.show()

    def plot_deriv_key(self, key, dt=None, show=False):
        # prints the simple deriv estimate of the data at key
        if key in self.deriv_dict:
            deriv_name = self.deriv_dict[key][1]
            plt.plot(self.data[deriv_name], label=deriv_name)
        else:
            assert dt is not None, "dt must be provided to plot the derivative if it wasn't provided before"
            plt.plot(np.diff(self.data[key]) / dt, label=f"d{key}/dt")
        if show:
            plt.legend()
            plt.show()

    def show_with_legend(self):
        plt.legend()
        leg = InteractiveLegend()
        plt.show()
def create_from_csv(file):
    #create a DataLogger from a csv file
    df = pd.read_csv(file)
    logger = DataLogger(file)
    for key in df.keys():
        logger.data[key] = df[key].tolist()
    return logger



class InteractiveLegend(object):
    def __init__(self, legend=None):
        if legend == None:
            legend = plt.gca().get_legend()
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.update()
    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))
        return lookup_artist, lookup_handle
    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()
    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()
    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()
