"use client";

import { useEffect } from "react";
import { AssistantRuntimeProvider, useAuiState } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";
import { useAppDispatch } from "@/context/app-context";

function MessageIdTracker() {
    const dispatch = useAppDispatch();
    const lastAssistantMessageId = useAuiState((s) => {
        const lastMsg = s.thread.messages[s.thread.messages.length - 1];
        return lastMsg?.role === "assistant" ? lastMsg.id : null;
    });

    useEffect(() => {
        if (lastAssistantMessageId) {
            dispatch({ type: "SET_CURRENT_MESSAGE_ID", messageId: lastAssistantMessageId });
        }
    }, [lastAssistantMessageId, dispatch]);

    return null;
}

export interface RagAreaProps {
    selectedSourceIds: string[];
    intentOverride?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onData: (dataPart: unknown) => void;
    onFinish: () => void;
}

export function RagArea({ selectedSourceIds, intentOverride, onData, onFinish }: RagAreaProps) {
    const runtime = useChatRuntime({
        transport: new AssistantChatTransport({
            api: "/api/chat",
            body: {
                data: {
                    citations_enabled: true,
                    source_ids: selectedSourceIds,
                    intent_override: intentOverride ?? "",
                },
            },
        }),
        onData: onData as any,
        onFinish: onFinish as any,
    });

    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <MessageIdTracker />
            <Thread />
        </AssistantRuntimeProvider>
    );
}
