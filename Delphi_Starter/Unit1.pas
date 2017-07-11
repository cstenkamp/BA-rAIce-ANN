unit Unit1;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, IniFiles, ShlObj, ComObj, ActiveX, StdCtrls, shellAPI, ExtCtrls;

type
  TForm1 = class(TForm)
    Edit1: TEdit;
    Label1: TLabel;
    Button1: TButton;
    Label2: TLabel;
    Edit2: TEdit;
    Label3: TLabel;
    Edit3: TEdit;
    Label4: TLabel;
    Edit4: TEdit;
    Edit5: TEdit;
    Label5: TLabel;
    Button2: TButton;
    ListBox1: TListBox;
    Label6: TLabel;
    Edit6: TEdit;
    Timer1: TTimer;
    ListBox2: TListBox;
    ListBox3: TListBox;
    ListBox4: TListBox;
    Button4: TButton;
    procedure FormCreate(Sender: TObject);
    procedure Edit1Change(Sender: TObject);
    procedure Edit2Change(Sender: TObject);
    procedure Edit4Change(Sender: TObject);
    procedure Edit5Change(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Edit3Change(Sender: TObject);
    procedure Edit6Change(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure Button4Click(Sender: TObject);
  private
    { Private-Deklarationen }
  public
    { Public-Deklarationen }
  end;
function ExtractNumber(const s: string): integer;

var
  Form1: TForm1;
  ini: TIniFile;
  iniFileName, gamePath, pythonPath, args, checkpointfolder: String;
  keepcheckpoints, restartevery: integer;

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

  procedure ListFileDir(Path: string; FileList: TStrings);
  var
    SR: TSearchRec;
  begin
    if FindFirst(Path + '*.*', faAnyFile, SR) = 0 then
    begin
      repeat
        if (SR.Attr <> faDirectory) then
        begin
          FileList.Add(SR.Name);
        end;
      until FindNext(SR) <> 0;
      FindClose(SR);
    end;
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
	  edit2.Text := pythonPath;
	  edit3.Text := args;
	  edit4.Text := inttostr(keepcheckpoints);
	  edit5.Text := inttostr(restartevery);
	  edit6.Text := checkpointfolder;       
    timer1.Interval := restartevery*60000;
	end;

	procedure TForm1.Edit1Change(Sender: TObject);
	begin
    gamePath := edit1.text;
	  ini.WriteString('All', 'gamePath', gamePath);
	end;

	procedure TForm1.Edit2Change(Sender: TObject);
begin
    pythonPath := edit2.Text;
	  ini.WriteString('All', 'pythonPath', pythonPath);
end;

procedure TForm1.Edit4Change(Sender: TObject);
begin
  keepcheckpoints := strtoint(edit4.text);
  ini.WriteInteger('All', 'keepcheckpoints', keepcheckpoints);
end;

procedure TForm1.Edit5Change(Sender: TObject);
begin
  restartevery := strtoint(edit5.text);
  ini.WriteInteger('All', 'restartevery', restartevery);
  timer1.Interval := restartevery*60000;
end;

procedure TForm1.Button2Click(Sender: TObject);
var
  i,j, currmax, currmaxind, relevantval: integer;
  schondrin : boolean;
begin
  ListBox1.Items.Clear; ListBox2.Items.Clear; ListBox3.Items.Clear; ListBox4.Items.Clear;

  ListFileDir(ExtractFilePath(pythonPath)+'/'+checkpointfolder+'/', ListBox1.Items);
  For i := 0 to ListBox1.Count-1 do begin
     ListBox2.Items[i] := inttostr(ExtractNumber(StringReplace(ListBox1.Items[i],ExtractFileExt(ListBox1.Items[i]),'',[rfReplaceAll])))
  end;
  For i := 0 to ListBox2.Count-1 do begin
    schondrin := False;
    For j := 0 to ListBox3.Count-1 do begin
       if ListBox2.Items[i] = ListBox3.Items[j] then schondrin := True
    end;
    if ListBox2.Items[i] = '0' then schondrin := True;
    if not schondrin then ListBox3.Items.Add(ListBox2.Items[i]);
  end;
  For i := 0 to ListBox3.Count-1 do ListBox4.Items.Add(ListBox3.Items[i]);
  
  For j := 0 to ListBox4.Count-1 do begin
	  currmax := 0; currmaxind := 0;
	  For i := 0 to ListBox4.Count-1 do begin
		 if strtoint(ListBox4.Items[i]) > 0 then begin
		   if strtoint(ListBox4.Items[i]) > currmax then begin
			 currmax := strtoint(ListBox4.Items[i]);
			 currmaxind := i;
		   end;
		 end;
	  end;
	  ListBox4.Items[currmaxind] := inttostr(-(j+1));
  end;

  For i := 1 to strtoint(edit4.text) do begin
    relevantval := 0;
    For j := 0 to ListBox4.Count-1 do if -strtoint(ListBox4.Items[j]) = i then relevantval := strtoint(ListBox3.Items[j]);
    For j := 0 to ListBox2.Count-1 do begin
      if (strtoint(ListBox2.Items[j]) = relevantval) or (strtoint(ListBox2.Items[j]) = 0) then begin
        ListBox1.Items[j] := ''
      end;
    end;
  end;

  For i := 0 to ListBox1.Count-1 do begin
    if ListBox1.Items[i] <> '' then begin
      DeleteFile(ExtractFilePath(pythonPath)+'/'+checkpointfolder+'/'+ListBox1.Items[i]);
    end;
  end;
end;


procedure TForm1.Edit3Change(Sender: TObject);
begin
    args := edit3.text;
	  ini.WriteString('All', 'args', args);
end;

procedure TForm1.Edit6Change(Sender: TObject);
begin
  checkpointfolder := edit6.text;
  ini.WriteString('All', 'checkpointfolder', checkpointfolder);
end;




 procedure TForm1.Button1Click(Sender: TObject);
 var
    command: String;
	begin
	  ShellExecute(handle, '', 'taskkill.exe','/f /im rAIce.exe','c:\windows\system32\', SW_HIDE);
	  ShellExecute(handle, '', 'taskkill.exe','/f /im cmd.exe','c:\windows\system32\', SW_HIDE);
	  ShellExecute(handle, '', 'taskkill.exe','/f /im python.exe','c:\windows\system32\', SW_HIDE);
	  Application.ProcessMessages; Sleep(2000); Application.ProcessMessages;
    command := '/C cd "'+ExtractFilePath(pythonPath)+'" && ' + 'python '+ExtractFileName(pythonPath)+' '+args;         
    ShellExecute(0, nil, 'cmd.exe', PChar(command), nil, SW_SHOW);
	  Application.ProcessMessages; Sleep(80*1000); Application.ProcessMessages;
	  ShellExecute(handle, nil, Pchar(gamePath), nil, nil, SW_SHOW);
	  Application.ProcessMessages; Sleep(2000); Application.ProcessMessages;
    PostKeyEx32(VK_RETURN, [], False);   
	  Application.ProcessMessages; Sleep(20*1000); Application.ProcessMessages;
    PostKeyEx32(VK_DOWN, [], False);          
	  Application.ProcessMessages; Sleep(500); Application.ProcessMessages;
    PostKeyEx32(VK_DOWN, [], False);           
	  Application.ProcessMessages; Sleep(500); Application.ProcessMessages;
    PostKeyEx32(VK_RETURN, [], False);
	  Application.ProcessMessages; Sleep(500); Application.ProcessMessages;
 end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin    
  Form1.Button2Click(Sender);
  Form1.Button1Click(Sender);
end;

function ExtractNumber(const s: string): integer;
var
  i: Integer;
  len: Integer;
  Start: Integer;
  isDigit: boolean;
begin
  len := Length(s);

  Start := 0;
  for i := 1 to len do begin
    try
      StrToInt(s[i]);
      isDigit := True;
    except
      isDigit := False;
    end;

    if isDigit then begin
      if Start=0 then begin
        Start := i;
      end;
    end else begin
      if Start<>0 then begin
        Result := strtoint(Copy(s, Start, i-Start));
        exit;
      end;
    end;
  end;
  if Start<>0 then begin
        Result := strtoint(Copy(s, Start, len));
        exit;
  end;
  Result := 0
end;


procedure TForm1.Button4Click(Sender: TObject);
begin
	  ShellExecute(handle, '', 'taskkill.exe','/f /im rAIce.exe','c:\windows\system32\', SW_HIDE);
	  ShellExecute(handle, '', 'taskkill.exe','/f /im cmd.exe','c:\windows\system32\', SW_HIDE);
	  ShellExecute(handle, '', 'taskkill.exe','/f /im python.exe','c:\windows\system32\', SW_HIDE);
	  Application.ProcessMessages; Sleep(2000); Application.ProcessMessages;
end;

Initialization
  iniFileName := ExtractFilePath(ParamStr(0)) + 'settings.ini';
  ini := TIniFile.Create(iniFileName);
  gamePath := ini.ReadString('All', 'gamePath', GetSystemPath(CSIDL_DESKTOPDIRECTORY)+'\rAIce.exe');
  pythonPath := ini.ReadString('All', 'pythonPath', 'C:\Users\csten_000\Documents\UNI\SEMESTER 7\Bachelorarbeit\BA-rAIce-ANN\server.py');
  args := ini.ReadString('All', 'args', '');
  checkpointfolder := ini.ReadString('All', 'checkpointfolder', 'RL_Learn');
  keepcheckpoints := ini.ReadInteger('All', 'keepcheckpoints', 2);
  restartevery := ini.ReadInteger('All', 'restartevery', 180);
Finalization
  ini.Free;
end.
 