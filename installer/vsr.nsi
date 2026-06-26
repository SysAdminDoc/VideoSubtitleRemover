; Video Subtitle Remover Pro -- NSIS installer (RM-51)
;
; This script wraps the PyInstaller --onedir build into a one-click
; installer. Run after the GitHub Actions build produces the
; `dist/VideoSubtitleRemoverPro/` directory. Outputs
; `VideoSubtitleRemoverPro-Setup.exe` in the working directory.
;
; The installer:
;   - Registers Start Menu + Desktop shortcuts.
;   - Adds an Uninstall entry in Add/Remove Programs.
;   - Installs a file-extension handler for the common video formats
;     so a double-click on `.mp4` etc. can route into VSR (RM-58).
;     The default verb stays "Open with..." in Windows; the verb
;     here is "Send to VSR".
;   - Refuses to install over a running instance.
;
; Build with: makensis installer/vsr.nsi

!define APPNAME      "Video Subtitle Remover Pro"
!define COMPANY      "SysAdminDoc"
!define APPID        "VideoSubtitleRemoverPro"
!define EXENAME      "VideoSubtitleRemoverPro.exe"

!define VERSIONMAJOR 3
!define VERSIONMINOR 17
!define VERSIONPATCH 2

Name "${APPNAME}"
OutFile "VideoSubtitleRemoverPro-Setup.exe"
InstallDir "$PROGRAMFILES64\${APPID}"
InstallDirRegKey HKLM "Software\${APPID}" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

VIProductVersion "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONPATCH}.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "CompanyName" "${COMPANY}"
VIAddVersionKey "FileDescription" "AI-powered subtitle remover."

!include "MUI2.nsh"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Application files (required)" SecCore
    SectionIn RO

    ; Refuse to install while the app is running.
    FindProcDLL::FindProc "${EXENAME}"
    Pop $R0
    IntCmp $R0 1 0 +3
        MessageBox MB_OK|MB_ICONSTOP "Close Video Subtitle Remover Pro before installing."
        Abort

    SetOutPath "$INSTDIR"
    File /r "..\dist\VideoSubtitleRemoverPro\*.*"

    WriteRegStr HKLM "Software\${APPID}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" \
        "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" \
        "DisplayIcon" "$\"$INSTDIR\${EXENAME}$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" \
        "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" \
        "Publisher" "${COMPANY}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}" \
        "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONPATCH}"
    WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Start menu + desktop shortcuts" SecShortcuts
    CreateDirectory "$SMPROGRAMS\${APPNAME}"
    CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\${EXENAME}"
    CreateShortCut "$SMPROGRAMS\${APPNAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"
    CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\${EXENAME}"
SectionEnd

Section "File extension handler" SecFileAssoc
    ; RM-58: register a "Send to VSR" verb on the common video extensions.
    ; We intentionally do NOT take over the default Open verb -- that
    ; would surprise users who associate .mp4 with Windows Media Player.
    !macro RegisterVideoVerb EXT
        WriteRegStr HKCR "${EXT}\shell\OpenWithVSR" "" "Send to Video Subtitle Remover"
        WriteRegStr HKCR "${EXT}\shell\OpenWithVSR\command" "" \
            "$\"$INSTDIR\${EXENAME}$\" $\"%1$\""
    !macroend

    !insertmacro RegisterVideoVerb ".mp4"
    !insertmacro RegisterVideoVerb ".avi"
    !insertmacro RegisterVideoVerb ".mkv"
    !insertmacro RegisterVideoVerb ".mov"
    !insertmacro RegisterVideoVerb ".wmv"
    !insertmacro RegisterVideoVerb ".flv"
    !insertmacro RegisterVideoVerb ".webm"
    !insertmacro RegisterVideoVerb ".m4v"
    !insertmacro RegisterVideoVerb ".mpeg"
    !insertmacro RegisterVideoVerb ".mpg"
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\${APPNAME}.lnk"
    RMDir /r "$SMPROGRAMS\${APPNAME}"

    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPID}"
    DeleteRegKey HKLM "Software\${APPID}"

    ; Drop the file-extension handlers we registered above.
    DeleteRegKey HKCR ".mp4\shell\OpenWithVSR"
    DeleteRegKey HKCR ".avi\shell\OpenWithVSR"
    DeleteRegKey HKCR ".mkv\shell\OpenWithVSR"
    DeleteRegKey HKCR ".mov\shell\OpenWithVSR"
    DeleteRegKey HKCR ".wmv\shell\OpenWithVSR"
    DeleteRegKey HKCR ".flv\shell\OpenWithVSR"
    DeleteRegKey HKCR ".webm\shell\OpenWithVSR"
    DeleteRegKey HKCR ".m4v\shell\OpenWithVSR"
    DeleteRegKey HKCR ".mpeg\shell\OpenWithVSR"
    DeleteRegKey HKCR ".mpg\shell\OpenWithVSR"

    RMDir /r "$INSTDIR"
SectionEnd
