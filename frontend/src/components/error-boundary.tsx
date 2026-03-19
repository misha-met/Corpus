"use client";

import React, { Component, ReactNode } from "react";

interface Props {
    children?: ReactNode;
}

interface State {
    hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false
    };

    public static getDerivedStateFromError(_: Error): State {
        return { hasError: true };
    }

    public componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    private handleRetry = () => {
        this.setState({ hasError: false });
    };

    public render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center p-4 border border-red-500/20 bg-red-500/10 rounded-md">
                    <p className="text-red-400 mb-2">Something went wrong.</p>
                    <button
                        className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-100 rounded text-sm transition-colors"
                        onClick={this.handleRetry}
                    >
                        Retry
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}
