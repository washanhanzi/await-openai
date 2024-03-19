pub trait EventDataParser<C1, C2, R> {
    //parse_data will return actionable data during the stream.
    //like tool call in openai API.
    fn parse_data(&mut self, data: &C1) -> Option<C2>;
    fn get_response(self) -> R;
}
