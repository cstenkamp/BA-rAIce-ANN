unit Unit1;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, IniFiles, ShlObj, ComObj, ActiveX, StdCtrls, shellAPI;

type
  TForm1 = class(TForm)
    Edit1: TEdit;
    Label1: TLabel;
    Button1: TButton;
    Button2: TButton;
    procedure FormCreate(Sender: TObject);
    procedure Edit1Change(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
  private
    { Private-Deklarationen }
  public
    { Public-Deklarationen }
  end;

var
  Form1: TForm1;
  ini: TIniFile;
  iniFileName, gamePath: String;

implementation

{$R *.dfm}

	//GET DESKTOP FOLDERS
	function GetSystemPath(Folder: Integer): string;
	var
	  PIDL: PItemIDList;
	  Path: LPSTR;
	  AMalloc: IMalloc;
	begin
	  Path := StrAlloc(MAX_PATH);
	  SHGetSpecialFolderLocation(Application.Handle, Folder, PIDL);

	  if SHGetPathFromIDList(PIDL, Path) then Result := Path;

	  SHGetMalloc(AMalloc);
	  AMalloc.Free(PIDL);
	  StrDispose(Path);
	end;

  procedure PostKeyEx32(key: Word; const shift: TShiftState; specialkey: Boolean);
	{************************************************************
	* Procedure PostKeyEx32
	*
	* Parameters:
	*  key    : virtual keycode of the key to send. For printable
	*           keys this is simply the ANSI code (Ord(character)).
	*  shift  : state of the modifier keys. This is a set, so you
	*           can set several of these keys (shift, control, alt,
	*           mouse buttons) in tandem. The TShiftState type is
	*           declared in the Classes Unit.
	*  specialkey: normally this should be False. Set it to True to
	*           specify a key on the numeric keypad, for example.
	* Description:
	*  Uses keybd_event to manufacture a series of key events matching
	*  the passed parameters. The events go to the control with focus.
	*  Note that for characters key is always the upper-case version of
	*  the character. Sending without any modifier keys will result in
	*  a lower-case character, sending it with [ssShift] will result
	*  in an upper-case character!
	// Code by P. Below
	************************************************************}
	type
	  TShiftKeyInfo = record
		shift: Byte;
		vkey: Byte;
	  end;
	  byteset = set of 0..7;
	const
	  shiftkeys: array [1..3] of TShiftKeyInfo =
		((shift: Ord(ssCtrl); vkey: VK_CONTROL),
		(shift: Ord(ssShift); vkey: VK_SHIFT),
		(shift: Ord(ssAlt); vkey: VK_MENU));
	var
	  flag: DWORD;
	  bShift: ByteSet absolute shift;
	  i: Integer;
	begin
	  for i := 1 to 3 do
	  begin
		if shiftkeys[i].shift in bShift then
		  keybd_event(shiftkeys[i].vkey, MapVirtualKey(shiftkeys[i].vkey, 0), 0, 0);
	  end; { For }
	  if specialkey then
		flag := KEYEVENTF_EXTENDEDKEY
	  else
		flag := 0;

	  keybd_event(key, MapvirtualKey(key, 0), flag, 0);
	  flag := flag or KEYEVENTF_KEYUP;
	  keybd_event(key, MapvirtualKey(key, 0), flag, 0);

	  for i := 3 downto 1 do
	  begin
		if shiftkeys[i].shift in bShift then
		  keybd_event(shiftkeys[i].vkey, MapVirtualKey(shiftkeys[i].vkey, 0),
			KEYEVENTF_KEYUP, 0);
	  end; { For }
	end; { PostKeyEx32 }

  {...............................................................................................}


	procedure TForm1.FormCreate(Sender: TObject);
	begin
	  edit1.Text := gamePath;
	end;

	procedure TForm1.Edit1Change(Sender: TObject);
	begin
	  ini.WriteString('Paths', 'gamePath', edit1.text);
	end;

	procedure TForm1.Button1Click(Sender: TObject);
	begin
	  ShellExecute(handle, '', 'taskkill.exe','/f /im rAIce.exe','c:\windows\system32\', SW_HIDE);
	  Sleep(2000); Application.ProcessMessages;
	  ShellExecute(handle, nil, Pchar(gamePath), nil, nil, SW_SHOW);
	  Sleep(2000); Application.ProcessMessages;
    PostKeyEx32(VK_RETURN, [], False);
	end;



  procedure TForm1.Button2Click(Sender: TObject);
  begin
	  ShellExecute(handle, '', 'taskkill.exe','/f /im cmd.exe','c:\windows\system32\', SW_HIDE);
	  Sleep(2000); Application.ProcessMessages;

  end;



Initialization
  iniFileName := ExtractFilePath(ParamStr(0)) + 'settings.ini';
  ini := TIniFile.Create(iniFileName);
  gamePath := ini.ReadString('Paths', 'gamePath', GetSystemPath(CSIDL_DESKTOPDIRECTORY)+'\rAIce.exe');
Finalization
  ini.Free;
end.
 