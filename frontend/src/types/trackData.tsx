import { Feature } from "../orval/models/backend-api";

export interface TrackData {
  id: string;
  name: string;
  instrument: string;
  wavData: Blob;
  features: Feature;
  muted: boolean;
  volume: number;
}
