# Progress Visualization in Job Extraction

## Overview

The job extraction script (`extract_jobs.py`) now includes real-time progress visualization using `tqdm` progress bars, which work seamlessly in both sequential and multithreaded modes.

## Features

### Sequential Mode (--parallel 1 or default)

- Shows a progress bar with:
  - Number of jobs processed
  - Number of jobs currently queued
  - Percentage complete
  - Estimated time remaining
  - Processing rate (jobs/sec)

Example output:

```
Processing jobs: 45%|████▌     | 23/51 [00:12<00:15, 1.8 job/s, completed=23, queued=28]
```

### Parallel Mode (--parallel > 1)

- Shows a thread-safe progress bar with:
  - Number of jobs processed
  - Number of active worker threads
  - Total worker count
  - Percentage complete
  - Estimated time remaining
  - Processing rate (jobs/sec)

Example output:

```
Processing jobs: 67%|██████▋   | 34/51 [00:08<00:04, 4.2 job/s, completed=34, active=8, workers=8]
```

## Implementation Details

### Thread Safety

- The progress bar uses a dedicated lock (`pbar_lock`) to ensure thread-safe updates
- Updates are synchronized to prevent race conditions
- Dynamic total updates as child jobs are discovered

### Dynamic Job Discovery

- The progress bar's total count dynamically increases as child/descendant jobs are discovered
- This provides accurate completion estimates even for deeply nested pipeline hierarchies

### Error Handling

- Failed jobs still update the progress bar to maintain accurate counts
- Error details are logged while keeping the progress display clean

## Usage

The progress visualization is automatic and requires no additional flags:

```bash
# Sequential mode with progress
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json

# Parallel mode with progress (8 workers)
python -m extractor.extract_jobs --source config/source_config.json --output data/jobs.json --parallel 8
```

## Benefits

1. **Real-time feedback**: See extraction progress in real-time
2. **Performance monitoring**: Track jobs/second processing rate
3. **Better estimation**: Get ETA for large extractions
4. **Debugging aid**: Identify if processing has stalled
5. **Multithreading insights**: See active worker count and throughput

## Technical Notes

- Progress is written to stdout while logs go to stderr and log files
- The progress bar automatically adjusts to terminal width
- Child jobs discovered during processing increase the total dynamically
- Works correctly with the `--limit` and `--include` options
