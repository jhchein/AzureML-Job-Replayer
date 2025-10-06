# Troubleshooting: No Outputs in Replay Jobs

## Symptoms

- Replay jobs complete successfully
- No files appear in the Outputs tab
- `_replay_download_summary.json` is missing

## Diagnostic Steps

### 1. Check if the component actually ran

In Azure ML Studio, go to the replay job and check:

- **Status**: Should be "Completed" (not "Failed" or "Canceled")
- **Logs tab**: Look for `user_logs/std_log.txt` or similar

### 2. Review the execution logs

Look for these specific log messages in `user_logs/std_log.txt`:

```
Starting artifact download into local ./outputs ...
Manifest contains 4 folder prefix(es) to enumerate and download.
Container URL (with SAS): https://...
Source blob prefix for job: ExperimentRun/dcid.{job_name}
Listing blobs under prefix: 'ExperimentRun/dcid.{job_name}/outputs' ...
  ✓ Found X blob(s) under 'outputs'
```

### 3. Common Issues and Fixes

#### Issue: "Found 0 blob(s)" for all folders

**Cause**: Source blobs don't exist or path is wrong  
**Fix**:

- Verify original job had outputs: `az storage blob list --account-name {source_acct} --container-name azureml --prefix "ExperimentRun/dcid.{original_job_name}/outputs/" --auth-mode login`
- Check that extraction included `mlflow_artifact_paths` in the JSON

#### Issue: "Manifest missing required source fields"

**Cause**: SAS token generation failed or source config missing  
**Fix**:

- Ensure you ran build_pipeline with `--source config/source_config.json`
- Verify Azure CLI is authenticated: `az account show`
- Check you have "Storage Blob Data Reader" role on source storage account

#### Issue: "Artifact manifest disabled; skipping downloads"

**Cause**: Extraction ran with `--no-artifacts`  
**Fix**: Re-run extraction WITHOUT `--no-artifacts` flag

#### Issue: Component logs show authentication errors

**Cause**: SAS token expired or invalid permissions  
**Fix**:

- SAS tokens have 6-hour validity
- Don't wait too long between build_pipeline and job execution
- Ensure "Storage Blob Data Delegator" role on source storage

#### Issue: Downloads start but all fail

**Cause**: Network connectivity or permissions  
**Fix**:

- Check error details in `_replay_download_summary.json` (if created)
- Verify target workspace compute can reach source storage account
- Check firewall rules on source storage account

### 4. Manual Verification Commands

Check if source blobs exist:

```powershell
$job_name = "your_original_job_name_here"
$source_acct = "source_storage_account"

az storage blob list `
  --account-name $source_acct `
  --container-name azureml `
  --prefix "ExperimentRun/dcid.$job_name/outputs/" `
  --auth-mode login `
  --output table `
  --num-results 5
```

Check extracted JSON has artifact paths:

```powershell
$json = Get-Content data/jobs.json | ConvertFrom-Json
$job = $json | Where-Object { $_.name -eq $job_name }
$job.mlflow_artifact_paths
# Should show: ["outputs/", "system_logs/", "logs/", "user_logs/"]
```

### 5. Expected Log Output (Success)

```
Replaying metrics for job: original_job_xyz
Reading metrics from file: /mnt/azureml/.../metrics_abc.json
Successfully parsed metrics JSON from file. Found 5 metrics.
Starting artifact download into local ./outputs ...
Manifest contains 4 folder prefix(es) to enumerate and download.
Container URL (with SAS): https://sourceacct.blob.core.windows.net/azureml?<SAS_TOKEN>
Source blob prefix for job: ExperimentRun/dcid.original_job_xyz
Listing blobs under prefix: 'ExperimentRun/dcid.original_job_xyz/outputs' ...
  Folder prefix from manifest: 'outputs/'
  Cleaned folder: 'outputs'
  Full blob prefix: 'ExperimentRun/dcid.original_job_xyz/outputs'
    DEBUG: blob.name='ExperimentRun/dcid.original_job_xyz/outputs/model.pkl' -> rel_path='outputs/model.pkl'
  ✓ Found 12 blob(s) under 'outputs'
Listing blobs under prefix: 'ExperimentRun/dcid.original_job_xyz/system_logs' ...
  ✓ Found 3 blob(s) under 'system_logs'
Planned downloads: 15 blob file(s)
  SAMPLE SRC='outputs/model.pkl' -> DST='model.pkl'
Downloaded 15/15 files (bytes=12345678)
Artifact download summary: total=15 success=15 failed=0 bytes=12345678 time_sec=5.23
Wrote download summary to outputs/_replay_download_summary.json (will appear in Outputs + logs).
```

### 6. Test with Single Job

Try with a simple test:

```powershell
# Extract just one job
python -m extractor.extract_jobs `
  --source config/source_config.json `
  --output data/test_single.json `
  --include "single_job_name_here"

# Replay with debug
python -m replayer.build_pipeline `
  --input data/test_single.json `
  --target config/target_config.json `
  --source config/source_config.json `
  --limit 1
```

Then check the job logs carefully for any ERROR messages.

### 7. Enable More Verbose Logging

If still stuck, you can temporarily modify `log_metrics.py` to print the full manifest:

```python
print(f"DEBUG: Full manifest = {json.dumps(manifest, indent=2)}")
```

This will show exactly what configuration is being used.

## Quick Checklist

- [ ] Extraction ran successfully with artifact paths included
- [ ] Original jobs had outputs in source storage
- [ ] build_pipeline ran with both `--source` and `--copy-artifacts`
- [ ] Replay job completed successfully (not failed)
- [ ] Checked `user_logs/std_log.txt` in Azure ML Studio
- [ ] SAS token hasn't expired (within 6 hours)
- [ ] Azure CLI authenticated with correct subscription
- [ ] Sufficient RBAC permissions on both source and target storage
