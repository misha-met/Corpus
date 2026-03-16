"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Globe, Trash2 } from "lucide-react";
import { Map, Layer, Popup, Source } from "@vis.gl/react-maplibre";
import type { MapRef, MapLayerMouseEvent } from "@vis.gl/react-maplibre";
import type { Feature, FeatureCollection, Point } from "geojson";
import type { GeoJSONSource, LayerSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

import {
  sourceApi,
  type GeoMentionDetail,
  type GeoMentionGroup,
  type GeoMentionsResponse,
} from "@/lib/api-client";
import { useAppDispatch } from "@/context/app-context";

type GeoFeatureProperties = {
  place_name: string;
  geonameid: number;
  mention_count: number;
  max_confidence: number;
  source_ids: string[];
  chunk_ids: string[];
  matched_inputs: string[];
  mention_ids: string[];
  mentions: GeoMentionDetail[];
};

type HoverPopup = {
  lng: number;
  lat: number;
  placeName: string;
  mentionCount: number;
};

const CLUSTER_LAYER: LayerSpecification = {
  id: "clusters",
  source: "geo-mentions",
  type: "circle",
  filter: ["has", "point_count"],
  paint: {
    "circle-color": ["step", ["get", "point_count"], "#3b82f6", 5, "#1d4ed8", 20, "#1e3a8a"],
    "circle-radius": ["step", ["get", "point_count"], 18, 5, 26, 20, 34],
    "circle-opacity": 0.85,
  },
};

const CLUSTER_COUNT_LAYER: LayerSpecification = {
  id: "cluster-count",
  source: "geo-mentions",
  type: "symbol",
  filter: ["has", "point_count"],
  layout: {
    "text-field": "{point_count_abbreviated}",
    "text-size": 12,
    "text-font": ["Open Sans Bold"],
  },
  paint: { "text-color": "#ffffff" },
};

const UNCLUSTERED_LAYER: LayerSpecification = {
  id: "unclustered",
  source: "geo-mentions",
  type: "circle",
  filter: ["!", ["has", "point_count"]],
  paint: {
    "circle-color": ["interpolate", ["linear"], ["get", "max_confidence"], 0.75, "#f59e0b", 0.92, "#3b82f6"],
    "circle-opacity": ["interpolate", ["linear"], ["get", "max_confidence"], 0.75, 0.45, 0.92, 1.0],
    "circle-radius": 7,
    "circle-stroke-width": 1,
    "circle-stroke-color": "#ffffff",
  },
};

const DARK_MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

interface CorpusMapProps {
  onCountChange?: (count: number) => void;
  active?: boolean;
  refreshNonce?: number;
}

export function CorpusMap({
  onCountChange,
  active = true,
  refreshNonce = 0,
}: CorpusMapProps) {
  const dispatch = useAppDispatch();
  const mapRef = useRef<MapRef | null>(null);
  const hasFitBounds = useRef(false);

  const [data, setData] = useState<GeoMentionsResponse>({ count: 0, mentions: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoverPopup, setHoverPopup] = useState<HoverPopup | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<GeoMentionGroup | null>(null);
  const [deletingMentionId, setDeletingMentionId] = useState<string | null>(null);

  const loadMentions = useCallback(async (): Promise<GeoMentionsResponse> => {
    const res = await sourceApi.getGeoMentions(undefined, 0.75);
    setData(res);
    onCountChange?.(res.count);
    return res;
  }, [onCountChange]);

  useEffect(() => {
    if (!active) {
      return;
    }

    let cancelled = false;
    setSelectedGroup(null);
    setHoverPopup(null);
    setIsLoading(true);
    setError(null);

    loadMentions()
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load geocoded mentions");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [active, loadMentions, refreshNonce]);

  const geojson = useMemo<FeatureCollection<Point, GeoFeatureProperties>>(() => {
    const features: Array<Feature<Point, GeoFeatureProperties>> = data.mentions.map((group) => ({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [group.lon, group.lat],
      },
      properties: {
        place_name: group.place_name,
        geonameid: group.geonameid,
        mention_count: group.mention_count,
        max_confidence: group.max_confidence,
        source_ids: group.source_ids,
        chunk_ids: group.chunk_ids,
        matched_inputs: group.matched_inputs,
        mention_ids: group.mention_ids,
        mentions: group.mentions,
      },
    }));

    return {
      type: "FeatureCollection",
      features,
    };
  }, [data.mentions]);

  useEffect(() => {
    if (!selectedGroup) {
      return;
    }
    const next = data.mentions.find((item) => item.geonameid === selectedGroup.geonameid) ?? null;
    if (next === selectedGroup) {
      return;
    }
    setSelectedGroup(next);
  }, [data.mentions, selectedGroup]);

  useEffect(() => {
    if (hasFitBounds.current) {
      return;
    }
    if (!geojson.features.length) {
      return;
    }

    const map = mapRef.current?.getMap();
    if (!map) {
      return;
    }

    let minLon = Infinity;
    let minLat = Infinity;
    let maxLon = -Infinity;
    let maxLat = -Infinity;

    for (const feature of geojson.features) {
      const [lon, lat] = feature.geometry.coordinates;
      minLon = Math.min(minLon, lon);
      minLat = Math.min(minLat, lat);
      maxLon = Math.max(maxLon, lon);
      maxLat = Math.max(maxLat, lat);
    }

    if (!Number.isFinite(minLon) || !Number.isFinite(minLat) || !Number.isFinite(maxLon) || !Number.isFinite(maxLat)) {
      return;
    }

    map.fitBounds(
      [
        [minLon, minLat],
        [maxLon, maxLat],
      ],
      { padding: 60, duration: 900 },
    );
    hasFitBounds.current = true;
  }, [geojson.features]);

  const onMapClick = useCallback((evt: MapLayerMouseEvent) => {
    const first = evt.features?.[0];
    if (!first) {
      return;
    }

    const map = mapRef.current?.getMap();
    if (!map) {
      return;
    }

    const properties = (first.properties ?? {}) as Record<string, unknown>;
    const geometry = first.geometry;

    if (properties.point_count != null) {
      const clusterId = Number(properties.cluster_id);
      const source = map.getSource("geo-mentions") as GeoJSONSource | undefined;
      if (!source || Number.isNaN(clusterId)) {
        return;
      }

      source
        .getClusterExpansionZoom(clusterId)
        .then((zoom) => {
          if (!geometry || geometry.type !== "Point") {
            return;
          }
          map.easeTo({
            center: geometry.coordinates as [number, number],
            zoom,
            duration: 450,
          });
        })
        .catch(() => {
          // ignore cluster zoom errors
        });
      return;
    }

    const geonameid = Number(properties.geonameid);
    if (Number.isNaN(geonameid)) {
      return;
    }

    const group = data.mentions.find((item) => item.geonameid === geonameid) ?? null;
    setSelectedGroup(group);
  }, [data.mentions]);

  const onMapMouseMove = useCallback((evt: MapLayerMouseEvent) => {
    const first = evt.features?.[0];
    if (!first) {
      setHoverPopup(null);
      return;
    }

    const properties = (first.properties ?? {}) as Record<string, unknown>;
    if (properties.point_count != null) {
      setHoverPopup(null);
      return;
    }

    if (!first.geometry || first.geometry.type !== "Point") {
      setHoverPopup(null);
      return;
    }

    setHoverPopup({
      lng: first.geometry.coordinates[0],
      lat: first.geometry.coordinates[1],
      placeName: String(properties.place_name ?? ""),
      mentionCount: Number(properties.mention_count ?? 0),
    });
  }, []);

  const handleViewChunk = useCallback(async (sourceId: string, chunkId: string) => {
    const chunk = await sourceApi.getChunk(sourceId, chunkId);
    dispatch({
      type: "SET_ACTIVE_CITATION",
      citation: {
        number: 0,
        source_id: sourceId,
        chunk_id: chunkId,
        page: chunk.page_number ?? null,
        header_path: chunk.header_path,
        chunk_text: chunk.chunk_text,
      },
    });
  }, [dispatch]);

  const handleDeleteMention = useCallback(async (mentionId: string) => {
    if (!selectedGroup) {
      return;
    }

    setDeletingMentionId(mentionId);
    try {
      await sourceApi.deleteGeoMention(mentionId);
      const refreshed = await loadMentions();
      const updatedGroup = refreshed.mentions.find((m) => m.geonameid === selectedGroup.geonameid) ?? null;
      setSelectedGroup(updatedGroup);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete geo mention");
    } finally {
      setDeletingMentionId(null);
    }
  }, [loadMentions, selectedGroup]);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-400">
        Loading map...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-red-300">
        {error}
      </div>
    );
  }

  if (!data.count) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-gray-300">
        <Globe className="h-9 w-9 text-gray-500" />
        <p>No locations indexed yet - re-ingest a document with geotagging enabled</p>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: 12, latitude: 25, zoom: 1.7 }}
        mapStyle={DARK_MAP_STYLE}
        interactiveLayerIds={["clusters", "unclustered"]}
        onClick={onMapClick}
        onMouseMove={onMapMouseMove}
      >
        <Source
          id="geo-mentions"
          type="geojson"
          data={geojson}
          cluster={true}
          clusterMaxZoom={14}
          clusterRadius={50}
        >
          <Layer {...CLUSTER_LAYER} />
          <Layer {...CLUSTER_COUNT_LAYER} />
          <Layer {...UNCLUSTERED_LAYER} />
        </Source>

        {hoverPopup && (
          <Popup
            longitude={hoverPopup.lng}
            latitude={hoverPopup.lat}
            closeButton={false}
            closeOnClick={false}
            anchor="top"
            offset={10}
            className="pointer-events-none"
          >
            <div className="rounded-lg border border-white/15 bg-black/80 px-3 py-2 text-xs shadow-[0_10px_24px_rgba(0,0,0,0.45)] backdrop-blur-md">
              <p className="font-semibold text-white">{hoverPopup.placeName}</p>
              <p className="text-white/70">
                {hoverPopup.mentionCount} mention{hoverPopup.mentionCount === 1 ? "" : "s"}
              </p>
            </div>
          </Popup>
        )}
      </Map>

      <aside
        className={`absolute inset-y-0 right-0 z-30 w-[min(95%,26rem)] transform border-l border-gray-800 bg-gray-950/82 backdrop-blur-xl transition-transform duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${
          selectedGroup ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <div>
            <p className="text-sm font-semibold text-gray-100">{selectedGroup?.place_name ?? ""}</p>
            <p className="text-xs text-gray-400">
              {selectedGroup?.mention_count ?? 0} mention{(selectedGroup?.mention_count ?? 0) === 1 ? "" : "s"}
            </p>
          </div>
          <button
            className="rounded px-2 py-1 text-xs text-gray-300 hover:bg-white/10"
            onClick={() => setSelectedGroup(null)}
          >
            Close
          </button>
        </div>

        <div className="h-[calc(100%-56px)] overflow-y-auto px-3 py-3">
          {(selectedGroup?.mentions ?? []).map((mention) => (
            <div key={mention.id} className="mb-3 rounded-lg border border-gray-800 bg-gray-900/70 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="truncate text-xs font-medium text-gray-200">{mention.source_id}</p>
                <span className="rounded border border-blue-500/40 bg-blue-500/10 px-2 py-0.5 text-[10px] text-blue-200">
                  {(mention.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <p className="mb-3 text-xs text-gray-300">
                mentioned as &apos;{mention.matched_input}&apos;
              </p>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleViewChunk(mention.source_id, mention.chunk_id)}
                  className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-200 hover:border-gray-500 hover:bg-white/10"
                >
                  -&gt; view
                </button>
                <button
                  type="button"
                  onClick={() => handleDeleteMention(mention.id)}
                  disabled={deletingMentionId === mention.id}
                  className="rounded border border-red-500/30 p-1 text-red-300 hover:bg-red-500/10 disabled:opacity-50"
                  title="Delete mention"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
