import inspect
from functools import wraps  # , WRAPPER_ASSIGNMENTS
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union

from flask import Response, current_app, jsonify, make_response, request
from pydantic import BaseModel, ValidationError
from pydantic.tools import parse_obj_as

from .converters import convert_query_params
from .exceptions import (
    InvalidIterableOfModelsException,
    JsonBodyParsingError,
    ManyModelValidationError,
    UnsupportedMediaType,
)

try:
    from flask_restful import original_flask_make_response as make_response
except ImportError:
    pass


def make_json_response(
    content: Union[BaseModel, Iterable[BaseModel]],
    status_code: int,
    by_alias: bool,
    exclude_none: bool = False,
    many: bool = False,
) -> Response:
    """serializes model, creates JSON response with given status code"""
    if many:
        s = ", ".join(
            [
                model.json(exclude_none=exclude_none, by_alias=by_alias)
                for model in content
            ]
        )
        js = f"[{s}]"
    else:
        js = content.json(exclude_none=exclude_none, by_alias=by_alias)
    response = make_response(js, status_code)
    response.mimetype = "application/json"
    return response


def unsupported_media_type_response(request_cont_type: str) -> Response:
    body = {
        "detail": f"Unsupported media type '{request_cont_type}' in request. "
        "'application/json' is required."
    }
    return make_response(jsonify(body), 415)


def is_iterable_of_models(content: Any) -> bool:
    try:
        return all(isinstance(obj, BaseModel) for obj in content)
    except TypeError:
        return False


def parse_custom_root_type(model: Type[BaseModel], obj):
    return model.parse_obj(obj).__root__


def has_custom_root_type(model: Type[BaseModel], obj):
    return isinstance(obj, list) and "__root__" in model.__fields__


def validate_many_models(model: Type[BaseModel], content: Any) -> List[BaseModel]:
    try:
        return [model(**fields) for fields in content]
    except TypeError:
        # iteration through `content` fails
        err = [
            {
                "loc": ["root"],
                "msg": "is not an array of objects",
                "type": "type_error.array",
            }
        ]
        raise ManyModelValidationError(err)
    except ValidationError as ve:
        raise ManyModelValidationError(ve.errors())


def validate_path_params(func: Callable, kwargs: dict) -> Tuple[dict, list]:
    errors = []
    validated = {}
    for name, type_ in func.__annotations__.items():
        if name in {"query", "body", "return"}:
            continue
        try:
            value = parse_obj_as(type_, kwargs.get(name))
            validated[name] = value
        except ValidationError as e:
            err = e.errors()[0]
            err["loc"] = [name]
            errors.append(err)
    kwargs = {**kwargs, **validated}
    return kwargs, errors


class ValidatedViewFunc(object):

    view_func: Callable[..., Any]
    body_model: Optional[Type[BaseModel]]
    query_model: Optional[Type[BaseModel]]

    def __init__(
        self,
        view_func: Callable[..., Any],
        body: Optional[Type[BaseModel]] = None,
        query: Optional[Type[BaseModel]] = None,
        on_success_status: int = 200,
        exclude_none: bool = False,
        response_many: bool = False,
        request_body_many: bool = False,
        response_by_alias: bool = False,
        include_in_schema: bool = True,
    ):
        self.view_func = view_func
        # TODO: test these

        self.body_model = view_func.__annotations__.get("body") or body
        self.body_model_in_kwargs = (
            "body" in inspect.signature(view_func).parameters.keys()
        )

        self.query_model = view_func.__annotations__.get("query") or query
        self.query_model_in_kwargs = (
            "query" in inspect.signature(view_func).parameters.keys()
        )

        self.request_body_many = request_body_many

        self.get_query_model = self._query_model_formatter()
        self.get_body_model = self._body_model_formatter()

        self.on_success_status = on_success_status
        self.exclude_none = exclude_none
        self.response_many = response_many
        self.request_body_many = request_body_many
        self.response_by_alias = response_by_alias
        self.include_in_schema = include_in_schema

        wraps(view_func)(self)

    @property
    def __include_in_schema__(self):
        return self.include_in_schema

    def _body_model_formatter(self) -> Callable[[dict], tuple]:
        """
        Predetermine how we are supposed to call the validation for the body
        model.
        """
        if self.body_model is None:

            def get_body_model(body_params: dict):
                return None, None

        elif "__root__" in self.body_model.__fields__:

            def get_body_model(body_params: dict):
                try:
                    data = self.body_model(__root__=body_params).__root__
                    return data, None
                except ValidationError as ve:
                    return None, ve.errors()

        elif self.request_body_many:

            def get_body_model(body_params: dict):
                try:
                    data = validate_many_models(self.body_model, body_params)
                    return data, None
                except ManyModelValidationError as e:
                    return None, e.errors()

        else:

            def get_body_model(body_params: dict):
                try:
                    return self.body_model(**body_params), None
                except TypeError:
                    content_type = request.headers.get("Content-Type", "").lower()
                    media_type = content_type.split(";")[0]
                    if media_type != "application/json":
                        raise UnsupportedMediaType(content_type)
                    else:
                        raise JsonBodyParsingError()
                except ValidationError as ve:
                    return None, ve.errors()

        return get_body_model

    def _query_model_formatter(self):
        if self.query_model is None:

            def get_query_model(query_params: dict):
                return None, None

        else:

            def get_query_model(query_params: dict):
                query_params = convert_query_params(query_params, self.query_model)
                try:
                    q = self.query_model(**query_params)
                except ValidationError as ve:
                    return None, ve.errors()
                else:
                    return q, None

        return get_query_model

    def __call__(self, *args, **kwargs):
        try:
            kwargs, path_errs = validate_path_params(self.view_func, kwargs)
            query, query_errs = self.get_query_model(request.args)
            body, body_errs = self.get_body_model(request.get_json())
        except UnsupportedMediaType as e:
            return unsupported_media_type_response(e.args[0])

        request.body_params = body
        if self.body_model_in_kwargs:
            kwargs["body"] = body

        request.query_params = query
        if self.query_model_in_kwargs:
            kwargs["query"] = query

        if any([path_errs, query_errs, body_errs]):
            status_code = current_app.config.get(
                "FLASK_PYDANTIC_VALIDATION_ERROR_STATUS_CODE", 400
            )
            errors = {}
            if path_errs:
                errors["path_params"] = path_errs
            if query_errs:
                errors["query_params"] = query_errs
            if body_errs:
                errors["body_params"] = body_errs
            res = ValidationErrorResponse(validation_error=errors)

            return make_response(jsonify(res.dict(exclude_none=True)), status_code)

        res = self.view_func(*args, **kwargs)

        if self.response_many:
            if is_iterable_of_models(res):
                return make_json_response(
                    res,
                    self.on_success_status,
                    by_alias=self.response_by_alias,
                    exclude_none=self.exclude_none,
                    many=True,
                )
            else:
                raise InvalidIterableOfModelsException(res)

        if isinstance(res, BaseModel):
            return make_json_response(
                res,
                self.on_success_status,
                exclude_none=self.exclude_none,
                by_alias=self.response_by_alias,
            )

        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], BaseModel):
            return make_json_response(
                res[0],
                res[1],
                exclude_none=self.exclude_none,
                by_alias=self.response_by_alias,
            )

        return res


class ValidationErrorSummary(BaseModel):
    path_params: Any
    query_params: Any
    body_params: Any


class ValidationErrorResponse(BaseModel):
    validation_error: ValidationErrorSummary


def validate(
    body: Optional[Type[BaseModel]] = None,
    query: Optional[Type[BaseModel]] = None,
    on_success_status: int = 200,
    exclude_none: bool = False,
    response_many: bool = False,
    request_body_many: bool = False,
    response_by_alias: bool = False,
    include_in_schema: bool = True,
):
    """
    Decorator for route methods which will validate query and body parameters
    as well as serialize the response (if it derives from pydantic's BaseModel
    class).

    Request parameters are accessible via flask's `request` variable:
        - request.query_params
        - request.body_params

    Or directly as `kwargs`, if you define them in the decorated function.

    `exclude_none` whether to remove None fields from response
    `response_many` whether content of response consists of many objects
        (e. g. List[BaseModel]). Resulting response will be an array of
        serialized models.
    `request_body_many` whether response body contains array of given model
        (request.body_params then contains list of models i. e.
        List[BaseModel])

    example::

        from flask import request
        from flask_pydantic import validate
        from pydantic import BaseModel

        class Query(BaseModel):
            query: str

        class Body(BaseModel):
            color: str

        class MyModel(BaseModel):
            id: int
            color: str
            description: str

        ...

        @app.route("/")
        @validate(query=Query, body=Body)
        def test_route():
            query = request.query_params.query
            color = request.body_params.query

            return MyModel(...)

        @app.route("/kwargs")
        @validate()
        def test_route_kwargs(query:Query, body:Body):

            return MyModel(...)

    -> that will render JSON response with serialized MyModel instance
    """

    def decorate(func: Callable) -> Callable:

        return ValidatedViewFunc(
            view_func=func,
            body=body,
            query=query,
            on_success_status=on_success_status,
            exclude_none=exclude_none,
            response_many=response_many,
            request_body_many=request_body_many,
            response_by_alias=response_by_alias,
            include_in_schema=include_in_schema,
        )

    return decorate


# class FlaskPydantic(object):
#     app: Optional["Flask"]
#
#     def __init__(self, app=None, url="/docs"):
#         if app is not None:
#             self.init_app(app)
#
#     def init_app(self, app):
#         self.app = app
#         app.register_blueprint(self.blueprint())
#
#     def blueprint(self):
#         for k, view_func in self.app.view_functions.items():
#             if getattr(view_func, "__include_in_schema__", False):
#                 print("include me!")
