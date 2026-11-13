param([string]$Pptx, [string]$Pdf, [string]$PngDir)
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
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
