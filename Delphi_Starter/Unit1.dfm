object Form1: TForm1
  Left = 317
  Top = 261
  Width = 767
  Height = 272
  Caption = 'Form1'
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  OnCreate = FormCreate
  DesignSize = (
    759
    241)
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 2
    Top = 4
    Width = 99
    Height = 25
    Caption = 'rAIce-Pfad:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Label2: TLabel
    Left = 2
    Top = 36
    Width = 143
    Height = 25
    Caption = 'PythonFile Pfad:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Label3: TLabel
    Left = 2
    Top = 68
    Width = 101
    Height = 25
    Caption = 'Arguments:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Label4: TLabel
    Left = 2
    Top = 156
    Width = 290
    Height = 25
    Caption = 'Keep only newest x Checkpoints:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Label5: TLabel
    Left = 450
    Top = 156
    Width = 180
    Height = 25
    BiDiMode = bdRightToLeft
    Caption = 'Restart all x Minutes:'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentBiDiMode = False
    ParentFont = False
  end
  object Label6: TLabel
    Left = 2
    Top = 116
    Width = 172
    Height = 25
    Caption = 'Checkpoint-Folder: '
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -20
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
  end
  object Edit1: TEdit
    Left = 104
    Top = 9
    Width = 653
    Height = 24
    Anchors = [akLeft, akTop, akRight]
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 0
    OnChange = Edit1Change
  end
  object Button1: TButton
    Left = 416
    Top = 197
    Width = 131
    Height = 33
    Anchors = [akRight, akBottom]
    Caption = 'Restart now'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 1
    OnClick = Button1Click
  end
  object Edit2: TEdit
    Left = 144
    Top = 41
    Width = 613
    Height = 24
    Anchors = [akLeft, akTop, akRight]
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 2
    OnChange = Edit2Change
  end
  object Edit3: TEdit
    Left = 104
    Top = 73
    Width = 653
    Height = 24
    Anchors = [akLeft, akTop, akRight]
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 3
    OnChange = Edit3Change
  end
  object Edit4: TEdit
    Left = 298
    Top = 161
    Width = 95
    Height = 24
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 4
    OnChange = Edit4Change
  end
  object Edit5: TEdit
    Left = 634
    Top = 161
    Width = 111
    Height = 24
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 5
    OnChange = Edit5Change
  end
  object Button2: TButton
    Left = 184
    Top = 197
    Width = 225
    Height = 33
    Anchors = [akRight, akBottom]
    Caption = 'Delete Checkpoints now'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 6
    OnClick = Button2Click
  end
  object ListBox1: TListBox
    Left = 8
    Top = 200
    Width = 17
    Height = 25
    ItemHeight = 13
    TabOrder = 7
    Visible = False
  end
  object Edit6: TEdit
    Left = 168
    Top = 121
    Width = 217
    Height = 24
    Anchors = [akLeft, akTop, akRight]
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -13
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 8
    OnChange = Edit6Change
  end
  object ListBox2: TListBox
    Left = 32
    Top = 200
    Width = 17
    Height = 25
    ItemHeight = 13
    TabOrder = 9
    Visible = False
  end
  object ListBox3: TListBox
    Left = 56
    Top = 200
    Width = 17
    Height = 25
    ItemHeight = 13
    TabOrder = 10
    Visible = False
  end
  object ListBox4: TListBox
    Left = 80
    Top = 200
    Width = 17
    Height = 25
    ItemHeight = 13
    TabOrder = 11
    Visible = False
  end
  object Button4: TButton
    Left = 552
    Top = 197
    Width = 91
    Height = 33
    Anchors = [akRight, akBottom]
    Caption = 'Kill now'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -19
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 12
    OnClick = Button4Click
  end
  object ListBox5: TListBox
    Left = 104
    Top = 200
    Width = 17
    Height = 25
    ItemHeight = 13
    TabOrder = 13
    Visible = False
  end
  object Button5: TButton
    Left = 648
    Top = 197
    Width = 105
    Height = 33
    Anchors = [akRight, akBottom]
    Caption = 'Close Python'
    Font.Charset = DEFAULT_CHARSET
    Font.Color = clWindowText
    Font.Height = -16
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    TabOrder = 14
    OnClick = Button5Click
  end
  object Timer1: TTimer
    OnTimer = Timer1Timer
    Left = 128
    Top = 200
  end
end
