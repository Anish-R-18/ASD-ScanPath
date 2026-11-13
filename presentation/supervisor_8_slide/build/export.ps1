param([string]$Pptx, [string]$Pdf, [string]$PngDir)
# Exports the deck to PDF and per-slide PNGs through PowerPoint.
# PowerPoint is quit afterwards only if no other presentation is open,
# so a deck the user is editing is never closed.
New-Item -ItemType Directory -Force $PngDir | Out-Null
Get-ChildItem $PngDir -Filter "slide-*.png" | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly, Untitled, no window
    $p = $app.Presentations.Open($Pptx, -1, 0, 0)
    $p.SaveAs($Pdf, 32)   # ppSaveAsPDF
    $n = $p.Slides.Count
    for ($i = 1; $i -le $n; $i++) {
        $p.Slides.Item($i).Export((Join-Path $PngDir ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900)
    }
    $p.Close()
    "exported $n slides"
} finally {
    if ($app.Presentations.Count -eq 0) { $app.Quit() } else { "PowerPoint left running ($($app.Presentations.Count) other presentation(s) open)" }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
