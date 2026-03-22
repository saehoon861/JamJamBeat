declare module "onnxruntime-web" {
  export class Tensor {
    constructor(type: string, data: Float32Array, dims: readonly number[]);
    data: Float32Array | readonly number[];
    dims: readonly number[];
  }

  export namespace InferenceSession {
    interface SessionOptions {
      executionProviders?: string[];
    }
  }

  export class InferenceSession {
    static create(
      modelUrl: string,
      options?: InferenceSession.SessionOptions,
    ): Promise<InferenceSession>;
    inputNames: string[];
    outputNames: string[];
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }
}
