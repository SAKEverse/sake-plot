# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:53:48 2021

@author: gweiss01
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os
mpl.use('TKAgg')

class GridGraph:
    def __init__(self,path,filename,data):
        """
        Creates an object that stores tidy data from .csv that can create a dynamic facet plot.

        Parameters
        ----------
        path : str, Full path of the directory containing the data to be graphed.
        filename : str, name of the .csv file to export. 
        
        
        Returns
        -------
        None.

        """
        self.kind='box'
        self.first_time=True
        
        #pass inputs to object
        self.path=path
        self.filename=filename
        self.data = data
        #get the categories from the columns (exceptt the last one)
        
        self.param_list=list(self.data.columns[:-1])
        #the last column is the value to graph
        
        self.graph_value = self.data.columns[-1]
        
        #swtich the freuency to the fisrt value (hue)
        ind=self.param_list.index('freq')
        self.param_list[ind],self.param_list[0] = self.param_list[0],self.param_list[ind]
        self.pivot_params=self.param_list

        

    def on_pick(self,event):
        """
        Callback for clicking on graphs. Export data if title is clicked,
        and changes the category if graph parameter is clicked

        Parameters
        ----------
        event : matplotlib event object.

        Returns
        -------
        None.

        """
        pivot_params=self.param_list.copy()
        var1=''
        var2=''
        # if clicked on a graphing parameter
        if ":" in event.artist.get_text():
            switched=event.artist.get_text().split(":")[1]
            self.param_list.remove(switched)# put the clicked on at the end
            self.param_list.append(switched)
            self.draw_graph(self.kind)
            return
        # if clicked on a graph title
        elif '|' in event.artist.get_text():
            
                # parse the string for categories and variables
                str1,str2=event.artist.axes.get_title().split(r" | ")
                cat1,var1=str1.split(" = ")
                cat2,var2=str2.split(" = ")
                
                # create index by filtering for both variables
                index1=self.data[cat1]==var1
                index2=self.data[cat2]==var2
                export_index=index1&index2
                
                # update the list of cats to pivot back to
                pivot_params.remove(cat1)
                pivot_params.remove(cat2)
        elif " = " in event.artist.get_text():
                cat1,var1=event.artist.axes.get_title().split(" = ")
                export_index=self.data[cat1]==var1
                pivot_params.remove(cat1)
        else:
                export_index=np.ones(self.data.shape[0])==1
        
        
        all_data=pd.DataFrame()
        # loop through the variables in the second category
        for cond in self.data[pivot_params[1]].unique():
            
            # make new table with the filter index
            filtered=self.data[export_index & (self.data[pivot_params[1]]==cond)]
            
            # melt the table by the first category, creating a separate table for each var in the second category
            cond_df=filtered.pivot(columns=self.pivot_params[0],values=self.graph_value)
            cond_df=cond_df[self.data['freq'].unique()]
            cond_df = cond_df.transpose()
            cond_df.columns = [cond]*cond_df.shape[1]
            
            # add to a concatenated df
            all_data=pd.concat([all_data,cond_df],axis=1)
            
        # export to csv 
        save_path = os.path.join(self.path, self.filename.split(".")[0]+"_"+var1+"_"+var2+".csv")
        all_data.to_csv(save_path)
        print("-> Exported to:" + str(save_path) + "\n")

    
    def draw_graph(self,kind='box',params=None):
        """
        

        Parameters
        ----------
        kind : str, optional
            DESCRIPTION. Type of plot, eg. box, bar, violin, strip.
        params : list, optional
            DESCRIPTION. A list of 1-4 categories to graph with order:x,hue,col,row.

        Returns
        -------
        None.

        """
        self.kind=kind
        cats=['X:','Hue:','Col:','Row:']
        # pick the first 4 parameters
        if params != None: self.param_list = params
        if len(self.param_list) > 4:
            graph_params=self.param_list[:4]
        else:
            graph_params=self.param_list
        default=[None]*4
        for i,param in enumerate(graph_params):
            default[i]=param
        #graph the facet plot with the first 4 categories
        x,hue,col,row = default
        g=sns.catplot(data = self.data, x = x, y = self.graph_value, hue = hue, col = col, row = row, kind = kind,height=3.5,aspect=6/4)
        #make each plot title clickable
        axes=g.axes.flat
        for ax in axes:
            ax.set_title(ax.get_title(),picker=5)
        #add clickable options for x,hue,row,col
        spacing=np.linspace(.2,.9,4)
        g.tight_layout(pad=2)
        for i,text in enumerate(graph_params):
            plt.figtext(spacing[i],.01,cats[i]+text,fontsize=10,picker=5,color='blue',fontweight='bold')
        #add helpful notes to figure
        plt.figtext(.01,.01,"Click to change:")
        if len(self.param_list)>2:
            plt.figtext(.35,.97,"Click a graph title to export",fontsize=12,fontweight='bold')
        else:
            plt.figtext(.4,.97,"Click to export",fontsize=12,fontweight='bold',picker=5)
        #add the click callback to the figure
        g.fig.canvas.callbacks.connect('pick_event', self.on_pick)
        plt.show()


    
if __name__ == '__main__':
    path= r"C:\Users\gweiss01\Downloads\\"
    filename=r"melt_index.csv"
    data=pd.read_csv(os.path.join(path,filename),index_col=0)
    graph=GridGraph(path,filename,data)
    graph.draw_graph('violin')

