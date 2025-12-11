"""ui_styles.py"""

from config.app_config import get_config

def get_checkbox_style() -> str:
    """チェックボックスの統一スタイルを取得"""
    config = get_config()
    colors = config.get_ui_colors()

    surface = colors.get('surface', '#3c3c3c')
    surface_hover = colors.get('surface_hover', '#4c4c4c')
    checkbox_unchecked = colors.get('checkbox_unchecked', '#888888')
    checkbox_checked = colors.get('checkbox_checked', '#2196F3')
    checkbox_hover = colors.get('checkbox_hover', '#4c4c4c')
    info_hover = colors.get('info_hover', '#42A5F5')
    text_secondary = colors.get('text_secondary', '#aaaaaa')

    return f"""
        QCheckBox {{
            spacing: 5px;
            background-color: transparent;
            color: white;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {checkbox_unchecked};
            border-radius: 3px;
            background-color: {surface};
        }}
        QCheckBox::indicator:hover {{
            border: 2px solid {checkbox_checked};
            background-color: {checkbox_hover};
        }}
        QCheckBox::indicator:checked {{
            border: 2px solid {checkbox_checked};
            background-color: {checkbox_checked};
            image: url(none);
        }}
        QCheckBox::indicator:checked:hover {{
            border: 2px solid {info_hover};
            background-color: {info_hover};
        }}
        QCheckBox::indicator:unchecked {{
            border: 2px solid {checkbox_unchecked};
            background-color: {surface};
        }}
        QCheckBox::indicator:unchecked:hover {{
            border: 2px solid {text_secondary};
            background-color: {surface_hover};
        }}
    """


def get_radiobutton_style() -> str:
    """ラジオボタンの統一スタイルを取得"""
    config = get_config()
    colors = config.get_ui_colors()

    surface = colors.get('surface', '#3c3c3c')
    surface_hover = colors.get('surface_hover', '#4c4c4c')
    checkbox_unchecked = colors.get('checkbox_unchecked', '#888888')
    checkbox_checked = colors.get('checkbox_checked', '#2196F3')
    checkbox_hover = colors.get('checkbox_hover', '#4c4c4c')
    info_hover = colors.get('info_hover', '#42A5F5')
    text_secondary = colors.get('text_secondary', '#aaaaaa')

    return f"""
        QRadioButton {{
            spacing: 5px;
            background-color: transparent;
            color: white;
        }}
        QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {checkbox_unchecked};
            border-radius: 9px;
            background-color: {surface};
        }}
        QRadioButton::indicator:hover {{
            border: 2px solid {checkbox_checked};
            background-color: {checkbox_hover};
        }}
        QRadioButton::indicator:checked {{
            border: 2px solid {checkbox_checked};
            background-color: {checkbox_checked};
            image: url(none);
        }}
        QRadioButton::indicator:checked:hover {{
            border: 2px solid {info_hover};
            background-color: {info_hover};
        }}
        QRadioButton::indicator:unchecked {{
            border: 2px solid {checkbox_unchecked};
            background-color: {surface};
        }}
        QRadioButton::indicator:unchecked:hover {{
            border: 2px solid {text_secondary};
            background-color: {surface_hover};
        }}
    """


__all__ = ['get_checkbox_style', 'get_radiobutton_style']