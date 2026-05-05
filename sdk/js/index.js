/**
 * ML Platform JavaScript SDK
 * Works in Node.js and browsers.
 *
 * Usage:
 *   import { MLPlatform } from 'ml-platform-sdk';
 *   const client = new MLPlatform('http://localhost:8000', { apiKey: 'sk-xxx' });
 *   const ds = await client.createDataset('用戶點擊', { project: 'ecom' });
 *   await ds.push([{ page: '/home', clicks: 42 }]);
 *
 *   // Auto-flush stream
 *   const stream = client.stream('即時事件', { flushEvery: 5000 });
 *   stream.add({ event: 'click', ts: Date.now() });
 *   // call stream.close() when done
 */

class MLPlatform {
  constructor(baseUrl, { apiKey } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.headers = { 'Content-Type': 'application/json' };
    if (apiKey) this.headers['X-API-Key'] = apiKey;
  }

  async _fetch(path, opts = {}) {
    const url = `${this.baseUrl}${path}`;
    const resp = await fetch(url, { ...opts, headers: { ...this.headers, ...opts.headers } });
    if (!resp.ok) {
      const err = await resp.text().catch(() => '');
      throw new Error(`ML Platform API ${resp.status}: ${err}`);
    }
    return resp.json();
  }

  async createDataset(name, { project = 'default', data = [] } = {}) {
    const info = await this._fetch('/api/datasets/push', {
      method: 'POST',
      body: JSON.stringify({ name, project, data }),
    });
    return new Dataset(this, info.id, name, project);
  }

  async getDatasets(project) {
    const params = project ? `?project=${encodeURIComponent(project)}` : '';
    return (await this._fetch(`/api/datasets${params}`)).datasets;
  }

  async dataset(dsId) {
    const info = await this._fetch(`/api/datasets/${dsId}`);
    return new Dataset(this, dsId, info.name, info.project);
  }

  stream(name, { project = 'default', flushEvery = 5000, batchSize = 100 } = {}) {
    return new Stream(this, name, project, flushEvery, batchSize);
  }
}

class Dataset {
  constructor(client, id, name, project) {
    this.client = client;
    this.id = id;
    this.name = name;
    this.project = project;
  }

  async push(data) {
    return this.client._fetch(`/api/datasets/${this.id}/append`, {
      method: 'POST',
      body: JSON.stringify({ data }),
    });
  }

  async info() {
    return this.client._fetch(`/api/datasets/${this.id}`);
  }

  async preview(rows = 5) {
    return this.client._fetch(`/api/datasets/${this.id}/preview?rows=${rows}`);
  }
}

class Stream {
  constructor(client, name, project, flushEvery, batchSize) {
    this.client = client;
    this.name = name;
    this.project = project;
    this.flushEvery = flushEvery;
    this.batchSize = batchSize;
    this._buffer = [];
    this._dataset = null;
    this._timer = null;
    this._init = this._initialize();
  }

  async _initialize() {
    this._dataset = await this.client.createDataset(this.name, { project: this.project });
    this._timer = setInterval(() => this.flush(), this.flushEvery);
    return this._dataset;
  }

  add(row) {
    this._buffer.push(row);
    if (this._buffer.length >= this.batchSize) {
      this.flush();
    }
  }

  async flush() {
    if (!this._buffer.length) return;
    await this._init;
    const batch = this._buffer.splice(0);
    await this._dataset.push(batch);
  }

  async close() {
    if (this._timer) clearInterval(this._timer);
    await this.flush();
  }
}

// Export for both ESM and CJS
if (typeof module !== 'undefined') {
  module.exports = { MLPlatform, Dataset, Stream };
}
export { MLPlatform, Dataset, Stream };
