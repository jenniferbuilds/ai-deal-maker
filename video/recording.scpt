
tell application "QuickTime Player"
    activate
    new screen recording
    delay 1
    tell application "System Events"
        keystroke "r" using {command down}
    end tell
end tell
