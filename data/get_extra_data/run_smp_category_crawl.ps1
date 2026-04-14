$PYTHON_BIN = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$CRAWLER_SCRIPT = if ($env:CRAWLER_SCRIPT) { $env:CRAWLER_SCRIPT } else { ".\crawl_flickr_to_smp.py" }

$BASE_OUTPUT_DIR = if ($env:BASE_OUTPUT_DIR) { $env:BASE_OUTPUT_DIR } else { ".\category_crawls" }
$TOTAL_ITEMS = if ($env:TOTAL_ITEMS) { [int]$env:TOTAL_ITEMS } else { 100000 }
$LICENSES = if ($env:LICENSES) { $env:LICENSES } else { "4,5,7,8,9,10" }

$SLEEP_MIN = if ($env:SLEEP_MIN) { $env:SLEEP_MIN } else { "0.8" }
$SLEEP_MAX = if ($env:SLEEP_MAX) { $env:SLEEP_MAX } else { "2.0" }
$FLUSH_EVERY = if ($env:FLUSH_EVERY) { $env:FLUSH_EVERY } else { "200" }
$PER_PAGE = if ($env:PER_PAGE) { $env:PER_PAGE } else { "100" }
$SORT = if ($env:SORT) { $env:SORT } else { "date-posted-desc" }

# Category distribution aligned to test set
# Family base_target raised to 3000 (original ratio gives only 1120, too few)
$categories = @(
    @{ category="Travel&Active&Sports"; ratio=0.2518; query="travel";      base_target=25180 },
    @{ category="Holiday&Celebrations"; ratio=0.1079; query="holiday";     base_target=10790 },
    @{ category="Animal";               ratio=0.1023; query="animal";      base_target=10230 },
    @{ category="Entertainment";        ratio=0.0995; query="concert";     base_target=9950  },
    @{ category="Fashion";              ratio=0.0995; query="fashion";     base_target=9950  },
    @{ category="Whether&Season";       ratio=0.0827; query="winter";      base_target=8270  },
    @{ category="Social&People";        ratio=0.0800; query="people";      base_target=8000  },
    @{ category="Urban";                ratio=0.0665; query="city";        base_target=6650  },
    @{ category="Food";                 ratio=0.0655; query="food";        base_target=6550  },
    @{ category="Electronics";          ratio=0.0332; query="electronics"; base_target=3320  },
    @{ category="Family";               ratio=0.0112; query="family";      base_target=3000  }
)

function Sanitize-Name {
    param([string]$name)
    return $name.Replace("&", "_").Replace(" ", "_").Replace("/", "_")
}

function Get-ScaledTarget {
    param([int]$baseTarget, [int]$totalItems)
    return [Math]::Max(1, [Math]::Round($baseTarget * $totalItems / 100000.0))
}

New-Item -ItemType Directory -Force -Path $BASE_OUTPUT_DIR | Out-Null

Write-Host "Base output dir: $BASE_OUTPUT_DIR"
Write-Host "Total target items: $TOTAL_ITEMS"
Write-Host "Crawler script: $CRAWLER_SCRIPT"
Write-Host ""

foreach ($item in $categories) {
    $target = Get-ScaledTarget -baseTarget $item.base_target -totalItems $TOTAL_ITEMS
    $safeCategory = Sanitize-Name $item.category
    $outputDir = Join-Path $BASE_OUTPUT_DIR "extra_data_$safeCategory"

    Write-Host "========================================"
    Write-Host "Category : $($item.category)"
    Write-Host "Ratio    : $($item.ratio)"
    Write-Host "Query    : $($item.query)"
    Write-Host "Target   : $target"
    Write-Host "Output   : $outputDir"
    Write-Host "========================================"

    & $PYTHON_BIN $CRAWLER_SCRIPT `
        --output_dir $outputDir `
        --text $item.query `
        --max_items $target `
        --licenses $LICENSES `
        --download_images `
        --sleep_min $SLEEP_MIN `
        --sleep_max $SLEEP_MAX `
        --flush_every $FLUSH_EVERY `
        --per_page $PER_PAGE `
        --sort $SORT `
        --resume `
        --dedupe_on_image_path

    Write-Host ""
}

Write-Host "All category crawls finished."