import { config } from "../config";

/**
 * Minimal blob-storage interface so the bridge doesn't depend on a specific
 * provider. The default implementation writes to Google Cloud Storage (same GCP
 * project as the app, so it's covered by the GCP BAA and authed by the runtime
 * service account via ADC — no keys). Swapping to S3/R2 later is one new class.
 */
export interface BlobStore {
  /** Store `data` at `path`; returns a URI for logging. */
  put(path: string, data: Buffer, contentType: string): Promise<string>;
}

/**
 * Returns a GCS-backed store when recording is enabled and a bucket is set,
 * otherwise null (recording off). The GCS SDK is required lazily so it isn't
 * loaded when recording is disabled.
 */
export function createBlobStore(): BlobStore | null {
  if (!config.recording.enabled || !config.recording.bucket) return null;

  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { Storage } = require("@google-cloud/storage") as typeof import("@google-cloud/storage");
  const bucketName = config.recording.bucket;
  const bucket = new Storage().bucket(bucketName);

  return {
    async put(path, data, contentType): Promise<string> {
      await bucket.file(path).save(data, { contentType, resumable: false });
      return `gs://${bucketName}/${path}`;
    },
  };
}
