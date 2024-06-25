import flet as ft
from flet import (
    ElevatedButton,
    FilePicker,
    FilePickerResultEvent,
    Page,
    Row,
    Text,
    icons,
)
import os
import torch
import logging

from transparent_background.utils import get_backend
from transparent_background.Remover import entry_point

logging.basicConfig(level=logging.WARN)
logging.getLogger("flet_runtime").setLevel(logging.WARN)

options = {
    'output_type':'rgba',
    'mode':'base',
    'device':get_backend(),
    'r' : 0, 
    'g' : 0, 
    'b' : 0,
    'color' : [0, 0, 0],
    'ckpt':None,
    'threshold':None,
    'source':None,
    'dest':None,
    'use_custom':False,
    'jit':False
}

def main(page):
    def theme_changed(e):
        page.theme_mode = (
            ft.ThemeMode.DARK
            if page.theme_mode == ft.ThemeMode.LIGHT
            else ft.ThemeMode.LIGHT
        )
        page.update()

    def checkbox_changed(e):
        options['jit'] = jit_check.value
        page.update()

    def dropdown_changed(e):
        options['output_type'] = type_dropdown.value
        options['mode'] = mode_dropdown.value
        options['device'] = device_dropdown.value

        if options['output_type'] == 'custom' and not options['use_custom']:
            page.insert(1, ft.Row([r_field, g_field, b_field]))
            options['use_custom']=True

        elif options['output_type'] != 'custom' and options['use_custom']:
            options['use_custom']=False
            page.remove_at(1)

        output_text.value = 'Type: {}, Mode: {}, Device: {}'.format(options['output_type'], options['mode'], options['device'])
        page.update()

    def color_changed(e):
        options['r'] = int(r_field.value) if len(r_field.value) > 0 and r_field.value.isdigit() else 0
        options['g'] = int(g_field.value) if len(g_field.value) > 0 and g_field.value.isdigit() else 0
        options['b'] = int(b_field.value) if len(b_field.value) > 0 and b_field.value.isdigit() else 0
        options['color'] = [options['r'], options['g'], options['b']]
        output_text.value = 'Type: {}, Mode: {}, Device: {}'.format(options['output_type'], options['color'], options['device'])
        page.update()


    def pick_files_result(e: FilePickerResultEvent):
        file_path.update()
        options['source'] = e.files[0].path if e.files else 'Not Selected'
        file_path.value = options['source']
        file_path.update()
        if options['dest'] is None:
            options['dest'] = os.path.split(options['source'])[0]
            dest_path.value = options['dest']
            dest_path.update()

    # Open directory dialog
    def get_directory_result(e: FilePickerResultEvent):
        options['source'] = e.path if e.path else 'Not Selected'
        file_path.value = options['source']
        file_path.update()
        if options['dest'] is None:
            options['dest'] = os.path.split(options['source'])[0]
            dest_path.value = options['dest']
            dest_path.update()

    def get_dest_result(e: FilePickerResultEvent):
        options['dest'] = e.path if e.path else 'Not Selected'
        dest_path.value = options['dest']
        dest_path.update()

    def process(e):
        entry_point(options['output_type'], options['mode'], options['device'], options['ckpt'], options['source'], options['dest'], options['jit'], options['threshold'], progress_ring, page, preview, preview_out)

    page.window_width = 1000
    page.window_height = 600
    page.window_resizable = False

    page.theme_mode = ft.ThemeMode.LIGHT
    c = ft.Switch(label="Dark mode", on_change=theme_changed)

    output_text = ft.Text()
    jit_check = ft.Checkbox(label="use torchscript", value=False, on_change=checkbox_changed)

    type_dropdown = ft.Dropdown(
        label='type',
        width=200,
        hint_text='output type',
        on_change=dropdown_changed,
        options=[
            ft.dropdown.Option("rgba"),
            ft.dropdown.Option("map"),
            ft.dropdown.Option("green"),
            ft.dropdown.Option("white"),
            ft.dropdown.Option("blur"),
            ft.dropdown.Option("overlay"),
            ft.dropdown.Option("custom"),
        ],
    )
    type_dropdown.value=options['output_type']
    
    mode_dropdown = ft.Dropdown(
        label='mode',
        width=150,
        hint_text='mode',
        on_change=dropdown_changed,
        options=[
            ft.dropdown.Option("base"),
            ft.dropdown.Option("base-nightly"),
            ft.dropdown.Option("fast"),
        ],
    )
    mode_dropdown.value = options['mode']

    device_options = [ft.dropdown.Option("cpu")] 
    device_options += [ft.dropdown.Option("cuda:{}".format(i)) for i in range(torch.cuda.device_count())]
    device_options += ['mps:0'] if torch.backends.mps.is_available() else []

    device_dropdown = ft.Dropdown(
        label='device',
        width=150,
        hint_text='device',
        on_change=dropdown_changed,
        options=device_options
    )
    device_dropdown.value=options['device']

    r_field = ft.TextField(width=60, label='R', on_change=color_changed)
    g_field = ft.TextField(width=60, label='G', on_change=color_changed)
    b_field = ft.TextField(width=60, label='B', on_change=color_changed)

    r_field.value=str(options['r'])
    g_field.value=str(options['g'])
    b_field.value=str(options['b'])
    
    # ft.Image(src='figures/logo.png', width=100, height=100)

    page.add(
        ft.Row(
            [
                ft.Image(src='figures/logo.png', width=100, height=100),
                ft.Column(
                    [
                        ft.Row([c, output_text]),
                        ft.Row([type_dropdown, mode_dropdown, device_dropdown, jit_check])
                    ]
                )
            ]
        )
    )

    # page.add(ft.Row([c, output_text]))
    # page.add(ft.Row([type_dropdown, mode_dropdown, device_dropdown, jit_check]))

    pick_files_dialog = FilePicker(on_result=pick_files_result)

    get_directory_dialog = FilePicker(on_result=get_directory_result)
    file_path = Text()

    get_dest_dialog = FilePicker(on_result=get_dest_result)
    dest_path = Text()

    # hide all dialogs in overlay
    page.overlay.extend([pick_files_dialog, get_directory_dialog, get_dest_dialog])
    progress_ring = ft.ProgressRing(width=16, height=16, stroke_width = 2)
    progress_ring.value = 0

    preview = ft.Image(src=".preview.png", )
    preview_out = ft.Image(src=".preview_out.png")

    page.add(
        Row(
            [
                ElevatedButton(
                    "Open File",
                    icon=icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False
                    ),
                ),
                ElevatedButton(
                    "Open Directory",
                    icon=icons.FOLDER_OPEN,
                    on_click=lambda _: get_directory_dialog.get_directory_path(),
                    disabled=page.web,
                ),
                file_path,
            ]
        ),
        Row(
            [
                ElevatedButton(
                    "Open Destination",
                    icon=icons.FOLDER_OPEN,
                    on_click=lambda _: get_dest_dialog.get_directory_path(),
                    disabled=page.web,
                ),
                dest_path
            ]
        ),
        Row(
            [
                ElevatedButton(
                    "Process",
                    icon=icons.SEND,
                    on_click=process,
                    disabled=page.web,
                ),
                progress_ring
            ]
        ),
    )

    page.add(
        Row(
            [
                preview,
                preview_out
            ]
        )
    )

def gui():
    ft.app(target=main)

    if os.path.isfile('.preview.png'):
        os.remove('.preview.png')

    if os.path.isfile('.preview_out.png'):
        os.remove('.preview_out.png')

if __name__ == "__main__":
    gui()