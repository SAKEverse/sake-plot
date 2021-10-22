# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:53:48 2021

@author: gweiss01
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import PyQt5
# mpl.use('TkAgg')

class GridGraph:
    def __init__(self,path,filename,data):
        """
        Creates an object that stores tidy data from .csv that can create a dynamic facet plot.
        First column must be individual index.
        Last comlumn must be the values to graph (y-axis).
        Middle columns can contain any number of categories.

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
        self.g=None
        #pass inputs to object
        self.path=path
        self.filename=filename
        self.data = data
        #get the categories from the columns (exceptt the last one)
        
        self.param_list=list(self.data.columns[:-1])
        #the last column is the value to graph
        
        self.graph_value = self.data.columns[-1]
        

        PyQt5.QtCore.qInstallMessageHandler(self.handler)#supress the error message

        

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
            if 'X:' in event.artist.get_text(): return
            switched=event.artist.get_text().split(":")[1][1:]
            self.param_list.remove(switched)# put the clicked on at the end
            self.param_list.append(switched)
            exec(self.type)
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
            cond_df.columns = [cond+str(col) for col in cond_df.columns]
            
            # add to a concatenated df
            all_data=pd.concat([all_data,cond_df],axis=1)
            
        # export to csv 
        save_path = os.path.join(self.path, self.filename.split(".")[0]+"_"+var1+"_"+var2+".csv")
        all_data.to_csv(save_path)
        print("-> Exported to:" + str(save_path) + "\n")

    def make_interactive(self):
        cats=['X: ','Hue: ','Col: ','Row: ']
        #make each plot title clickable
        axes=self.g.axes.flat
        for ax in axes:
            ax.set_title(ax.get_title(),picker=5,fontsize=10)
        #add clickable options for x,hue,row,col
        spacing=np.linspace(.2,.8,4)
        self.g.tight_layout(pad=2)
        for i,text in enumerate(self.graph_params):
            plt.figtext(spacing[i],.01,cats[i]+text,fontsize=10,picker=5,color='blue',fontweight='bold')
        #add helpful notes to figure
        plt.figtext(.01,.01,"Click to change:")
        if len(self.param_list)>2:
            plt.figtext(.35,.97,"Click a graph title to export",fontsize=12,fontweight='bold')
        else:
            plt.figtext(.4,.97,"Click to export",fontsize=12,fontweight='bold',picker=5)
        #add the click callback to the figure
        self.g.fig.canvas.callbacks.connect('pick_event', self.on_pick)
        plt.show()
    
    def draw_graph(self,kind=None,params=None):
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
        #swtich the freuency to the first value (X)
        self.param_list.remove('freq')
        self.param_list = ['freq'] + self.param_list
        self.pivot_params=self.param_list
        
        self.type="self.draw_graph()"
        if kind: self.kind=kind
        # pick the first 4 parameters
        if params != None: self.param_list = params
        if len(self.param_list) > 4:
            self.graph_params=self.param_list[:4]
        else:
            self.graph_params=self.param_list
        default=[None]*4
        for i,param in enumerate(self.graph_params):
            default[i]=param
        #graph the facet plot with the first 4 categories
        x,hue,col,row = default
        height=2.5
        self.g=sns.catplot(data = self.data, x = x, y = self.graph_value, hue = hue, col = col, row = row, kind = self.kind,height=height,aspect=6/4,ci=68)
        self.make_interactive()
    
    def draw_psd(self,kind=False,params=None):
        """
        

        Parameters
        ----------
        kind : str, optional
            DESCRIPTION. Type of plot, eg. line.
        params : list, optional
            DESCRIPTION. A list of 1-4 categories to graph with order:x,hue,col,row.

        Returns
        -------
        None.

        """
        #swtich the freuency to the first value (X)
        self.param_list.remove('freq')
        self.param_list = ['freq'] + self.param_list
        self.pivot_params=self.param_list
        
        self.type="self.draw_psd()"
        # first pram stays 'freq', sets up to 3 other params
        if kind: self.kind=kind
        if params != None: self.param_list[1:] = params
        if len(self.param_list) > 4:
            self.graph_params=self.param_list[:4]
        else:
            self.graph_params=self.param_list
        default=[None]*4
        for i,param in enumerate(self.graph_params):
            default[i]=param
        #graph the facet plot with the first 4 categories
        x,hue,col,row = default
        self.g=sns.relplot(data = self.data, x = x, y = self.graph_value, hue = hue, col = col, row = row, height=2.5,aspect=6/4,kind='line',ci='se')
        self.make_interactive()

    def draw_dist(self,params=None):
        """
        

        Parameters
        ----------
        kind : str, optional
            DESCRIPTION. Type of plot, eg. line.
        params : list, optional
            DESCRIPTION. A list of 1-4 categories to graph with order:x,hue,col,row.

        Returns
        -------
        None.

        """
        #swtich the power to the fisrt value (X)
        self.param_list.remove('power')
        if 'threshold' in self.param_list:
            self.param_list.remove('threshold')
        self.param_list = ['power'] + self.param_list
        self.pivot_params=self.param_list
        
        self.type="self.draw_dist()"
        # first parameter 'value' becomes 'power', sets up to 3 other params
        
        
        if params != None: self.param_list[1:] = params
        if len(self.param_list) > 4:
            self.graph_params=self.param_list[:4]
        else:
            self.graph_params=self.param_list
        default=[None]*4
        for i,param in enumerate(self.graph_params):
            default[i]=param
        #graph the facet plot with the first 4 categories
        x,hue,col,row = default
        self.g=sns.relplot(data = self.data, x = x, y = self.graph_value, hue = hue, col = col, row = row, height=2.5,aspect=6/4,kind='line',ci='se')
        for axis in self.g.axes.flatten():
            old_title=axis.get_title()
            #get parameters from graph axis
            
            if ' | ' in old_title: 
                graph_dict={thing.split(" = ")[0]:thing.split(" = ")[1] for thing in old_title.split(" | ")}
                (key1,val1),(key2,val2)=graph_dict.items()
                temp=self.data[(self.data[key1]==val1) & (self.data[key2]==val2)]
            elif ' = ' in old_title:
                key1,val1=old_title.split(" = ")
                temp=self.data[(self.data[key1]==val1)]
            else:
                temp=self.data
                
            for line_name,line in zip(temp[hue].unique(),axis.lines):

                #get the threshold
                thresh=temp[temp[hue]==line_name]['threshold'].mean()
                thresh_loc=np.where((line.properties()['xdata']>thresh))[0][0]
                
                #Fill the areas under the curve to the left and right of the threshold
                hatch=''
                # if graph_dict['Genotype']=='Cre+':hatch='///'
                axis.fill_between(x=line.properties()['xdata'][thresh_loc:],
                                    y1=line.properties()['ydata'][thresh_loc:],
                                    y2=0,
                                    facecolor=line.properties()['color'],
                                    alpha=.3,
                                    hatch=hatch)
                
                # #find number of data points in the raw data below threshold and display on graph
                # this_data=temp[(temp[key1]==val1) & (temp[key2]==val2) & (temp['Metric']==test)]
                # total=len(temp[(temp[key1]==val1) & (temp[key2]==val2) & (temp['Metric']==test)]['Value'])
                # vuln=sum(temp[(temp[key1]==val1) & (temp[key2]==val2) & (temp['Metric']==test)]['Value'] < line.properties()['xdata'][thresh_loc])
                # axis.text(x=line.properties()['xdata'][thresh_loc],y=0,s="{:.0%}".format(vuln/total),ha='right',fontweight='bold',fontsize=24)
        self.make_interactive()
    

    def handler(msg_type, msg_log_context, msg_string, fourth_one):
        '''
        Supresses the error message with pass 

        Parameters
        ----------
        msg_type : 
        msg_log_context : 
        msg_string : 

        Returns
        -------
        None.

        '''
        pass


if __name__ == '__main__':
    path= r"C:\Users\gweiss01\Downloads\\"
    filename=r"melt_index.csv"
    data=pd.read_csv(os.path.join(path,filename),index_col=0)
    data2=pd.read_csv(r"C:\Users\gweiss01\Downloads\melted_dist.csv",index_col=0)
    
    # graph=GridGraph(path,filename,data)
    # graph.draw_graph('violin')

    graph=GridGraph(path,filename,data2)
    graph.draw_dist()

