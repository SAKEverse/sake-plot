# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:04:28 2021

@author: panton01
"""

import click

def file_present_check(ctx):

    # check if path exists
    if ctx.obj['index_present']:
        click.secho("\n -> Index file was not found.\n", fg = 'yellow', bold = True)
        return
    