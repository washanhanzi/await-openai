pub trait EventDataParser<C1, C2, R> {
    /// tool will return actionable data(like tool call) during the stream.
    /// For example, openai may return mutilple tool_calls during the stream, this method will return a complete tool_call when the parser can.
    /// We don't need to wait for the stream to complete to get all the tool_calls.
    fn tool(&mut self, data: &C1) -> Option<C2>;
    /// response will return the unaray response from the parsed stream data.
    fn response(self) -> R;
}
