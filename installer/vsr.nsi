; Video Subtitle Remover Pro -- NSIS installer (RM-51)
;
; This script wraps the PyInstaller --onedir build into a one-click
; installer. Run after the local PyInstaller build produces the
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
;
; REQUIRES NSIS >= 3.12. Earlier releases may use the Low IL temp directory
; while elevated, enabling a possible privilege escalation for installers
; running as SYSTEM. The compile-time guard fails older toolchains.

; NSIS_PACKEDVERSION allocates 8 bits to major, 12 bits to minor, 8 bits to
; revision, and 4 bits to build. 3.12 therefore packs to 0x0300C000.
; Fail compilation on anything older.
!ifdef NSIS_PACKEDVERSION
  !if ${NSIS_PACKEDVERSION} < 0x0300C000
    !error "NSIS >= 3.12 required (elevated Low IL temp hardening). Upgrade makensis."
  !endif
!else
  !warning "Cannot verify NSIS version; ensure makensis is >= 3.12."
!endif

!define APPNAME      "Video Subtitle Remover Pro"
!define COMPANY      "SysAdminDoc"
!define APPID        "VideoSubtitleRemoverPro"
!define EXENAME      "VideoSubtitleRemoverPro.exe"

!ifndef OUTPUT_DIR
  !define OUTPUT_DIR ".."
!endif
!ifndef DIST_DIR
  !define DIST_DIR "..\dist\VideoSubtitleRemoverPro"
!endif

!define VERSIONMAJOR 3
!define VERSIONMINOR 20
!define VERSIONPATCH 0

Name "${APPNAME}"
!ifdef VSR_SMOKE_BUILD
  ; Compile the identical application payload into a non-elevated harness so
  ; local release automation can prove extraction and frozen startup without
  ; changing the operator's Program Files or registry state.
  OutFile "${OUTPUT_DIR}\VideoSubtitleRemoverPro-Smoke-Setup.exe"
  InstallDir "$TEMP\${APPID}-Installer-Smoke"
  RequestExecutionLevel user
  SetCompress off
!else
  OutFile "${OUTPUT_DIR}\VideoSubtitleRemoverPro-Setup.exe"
  InstallDir "$PROGRAMFILES64\${APPID}"
  RequestExecutionLevel admin
  SetCompressor /SOLID lzma
!endif
!ifndef VSR_SMOKE_BUILD
  InstallDirRegKey HKLM "Software\${APPID}" "InstallDir"
!endif

VIProductVersion "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONPATCH}.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "CompanyName" "${COMPANY}"
VIAddVersionKey "FileDescription" "AI-powered subtitle remover."
VIAddVersionKey "FileVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONPATCH}.0"
VIAddVersionKey "LegalCopyright" "Copyright (c) ${COMPANY}"

!include "MUI2.nsh"
!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!ifndef VSR_SMOKE_BUILD
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES
!endif

!insertmacro MUI_LANGUAGE "English"

Section "Application files (required)" SecCore
    SectionIn RO

    !ifndef VSR_SMOKE_BUILD
        ; Refuse to install while the app is running.
        ; The app owns this named mutex for its process lifetime. System.dll
        ; ships with NSIS, avoiding an undeclared FindProcDLL dependency.
        System::Call 'kernel32::OpenMutexW(i 0x00100000, i 0, w "Local\VideoSubtitleRemoverPro.Running") p .R0'
        IntCmp $R0 0 app_not_running app_running app_running
        app_running:
            System::Call 'kernel32::CloseHandle(p r0)'
            MessageBox MB_OK|MB_ICONSTOP "Close Video Subtitle Remover Pro before installing."
            Abort
        app_not_running:
    !endif

    SetOutPath "$INSTDIR"
    File /r "${DIST_DIR}\*.*"

    !ifndef VSR_SMOKE_BUILD
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
    !endif
SectionEnd

!ifndef VSR_SMOKE_BUILD
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
!endif
